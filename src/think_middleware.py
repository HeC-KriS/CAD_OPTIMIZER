"""Middleware for handling DeepSeek-R1 reasoning models in LangChain.

Strips `<think>...</think>` Chain-of-Thought tags from model responses,
logs reasoning to the terminal, and isolates the JSON tool-calling payload
before it hits LangChain's tool execution framework to prevent parsing crashes.
Enforces single-tool-call per turn so DeepSeek-R1 can observe real outputs
from preceding steps before issuing subsequent actions.
"""

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import Field
from src.logger import TerminalLogger

# Regex matching <think>...</think> blocks across multiple lines
THINK_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK_REGEX = re.compile(r"<think>(.*)", flags=re.DOTALL | re.IGNORECASE)

# Regex matching markdown json blocks: ```json ... ```
JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def strip_think_tags(raw_text: str) -> Tuple[str, str]:
    """Strips <think>...</think> blocks from raw text.

    Returns:
        (clean_text, thought_content)
    """
    if not isinstance(raw_text, str):
        return "", ""

    thought_parts: List[str] = []

    def _replace_match(match: re.Match) -> str:
        thought_parts.append(match.group(1).strip())
        return ""

    clean_text = THINK_REGEX.sub(_replace_match, raw_text)

    if "<think>" in clean_text.lower():
        match_unclosed = UNCLOSED_THINK_REGEX.search(clean_text)
        if match_unclosed:
            unclosed_content = match_unclosed.group(1)
            json_search = re.search(r"(\{.*\})", unclosed_content, flags=re.DOTALL)
            if json_search:
                thought_parts.append(unclosed_content[: json_search.start()].strip())
                clean_text = json_search.group(1)
            else:
                thought_parts.append(unclosed_content.strip())
                clean_text = ""

    thought_content = "\n".join(thought_parts).strip()
    return clean_text.strip(), thought_content


def extract_all_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts tool calls from text (supporting single JSON, markdown blocks,

    or multiple JSON lines).
    """
    if not text:
        return []

    tool_calls: List[Dict[str, Any]] = []

    # 1. Check for markdown blocks
    for match in JSON_BLOCK_REGEX.finditer(text):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and any(k in parsed for k in ("name", "tool", "action")):
                tool_calls.append(parsed)
        except Exception:
            pass

    # 2. Match individual JSON objects on lines: {"name": ...}
    json_candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    for cand in json_candidates:
        try:
            parsed = json.loads(cand.strip())
            if isinstance(parsed, dict) and any(k in parsed for k in ("name", "tool", "action")):
                if parsed not in tool_calls:
                    tool_calls.append(parsed)
        except Exception:
            pass

    # 3. Fallback: try raw JSON substring if no candidates found
    if not tool_calls:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                parsed = json.loads(text[start_idx : end_idx + 1])
                if isinstance(parsed, dict):
                    tool_calls.append(parsed)
            except Exception:
                pass

    return tool_calls


def extract_target_file(messages: Optional[List[BaseMessage]]) -> str:
    """Finds the CAD target file from conversation messages."""
    if not messages:
        return "models/sample_bracket.step"

    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            match = re.search(r"'(models/[^']+)'", msg.content)
            if match:
                return match.group(1).strip("'\"")
            match = re.search(r"(models/[^\s'\"]+\.(?:step|stp|stl|dxf))", msg.content, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip("'\"")

    return "models/sample_bracket.step"


def determine_next_pipeline_call(
    messages: Optional[List[BaseMessage]],
) -> Optional[Dict[str, Any]]:
    """Determines the required next manufacturing tool if the LLM outputted
    conversational text instead of a JSON tool call before finishing the pipeline.
    """
    if not messages:
        return None

    target_file = extract_target_file(messages)
    base_name = os.path.splitext(os.path.basename(target_file))[0]
    oriented_stl = f"models/{base_name}_oriented.stl"

    tool_counts: Dict[str, int] = {}
    last_fea_data: Dict[str, Any] = {}

    for msg in messages:
        if hasattr(msg, "name") and msg.name:
            tool_counts[msg.name] = tool_counts.get(msg.name, 0) + 1
            if msg.name == "run_fea_pinn_analysis" and isinstance(getattr(msg, "content", None), str):
                try:
                    parsed = json.loads(msg.content)
                    if isinstance(parsed, dict):
                        last_fea_data = parsed
                except Exception:
                    pass
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_name = tc.get("name")
                if tc_name:
                    tool_counts[tc_name] = tool_counts.get(tc_name, 0) + 1

    # 1. If slicing already ran, the manufacturing pipeline is finished -> Allow final text summary!
    if tool_counts.get("slice_stl_to_gcode", 0) > 0:
        return None

    # 2. If auto_orient_stl ran, next MUST be slice_stl_to_gcode
    if tool_counts.get("auto_orient_stl", 0) > 0:
        return {
            "name": "slice_stl_to_gcode",
            "arguments": {
                "stl_filepath": oriented_stl,
                "layer_height_mm": 0.2,
                "fill_density": "20%",
            },
        }

    # 3. If FEA ran:
    if tool_counts.get("run_fea_pinn_analysis", 0) > 0:
        fea_good = last_fea_data.get("is_good", False)
        fea_iter = last_fea_data.get("iteration", tool_counts.get("run_fea_pinn_analysis", 1))
        max_iter = last_fea_data.get("max_iterations", 3)

        # If not compliant and loop limit not reached -> Re-run nTop with higher volume fraction
        if not fea_good and fea_iter < max_iter:
            next_iter = fea_iter + 1
            new_vol = round(0.22 + 0.08 * fea_iter, 2)
            return {
                "name": "run_ntop_topology_optimization",
                "arguments": {
                    "cad_filepath": target_file,
                    "target_volume_fraction": new_vol,
                    "iteration": next_iter,
                    "max_iterations": max_iter,
                },
            }
        else:
            # Compliant or loop limit reached -> Advance to auto-orientation
            return {
                "name": "auto_orient_stl",
                "arguments": {
                    "stl_filepath": target_file,
                    "overhang_angle_deg": 45.0,
                },
            }

    # 4. If nTop ran, next MUST be FEA analysis
    if tool_counts.get("run_ntop_topology_optimization", 0) > 0:
        current_iter = tool_counts.get("run_ntop_topology_optimization", 1)
        return {
            "name": "run_fea_pinn_analysis",
            "arguments": {
                "model_filepath": target_file,
                "iteration": current_iter,
                "target_fos": 1.5,
            },
        }

    # 5. If load_cad_model ran, next MUST be nTop iteration 1
    if tool_counts.get("load_cad_model", 0) > 0:
        return {
            "name": "run_ntop_topology_optimization",
            "arguments": {
                "cad_filepath": target_file,
                "target_volume_fraction": 0.3,
                "iteration": 1,
                "max_iterations": 3,
            },
        }

    # 6. Initial step: load_cad_model
    return {
        "name": "load_cad_model",
        "arguments": {
            "filepath": target_file,
        },
    }


def sanitize_ai_message(
    message: AIMessage,
    messages: Optional[List[BaseMessage]] = None,
    is_thinking_model: bool = False,
) -> AIMessage:
    """Sanitizes an AIMessage by optionally stripping <think> tags (if a thinking model),
    logging the reasoning trace, and parsing raw JSON tool calls.
    Enforces sequential single-step tool execution per turn and auto-recovers
    from premature conversational text exits before pipeline completion.
    """
    raw_content = message.content if isinstance(message.content, str) else ""

    if is_thinking_model:
        clean_content, thought_content = strip_think_tags(raw_content)
        if thought_content:
            TerminalLogger.thought(thought_content)
    else:
        # Standard model mode: no think tags
        clean_content = raw_content

    message.content = clean_content

    # If the model emitted tool calls in text rather than message.tool_calls
    if not message.tool_calls and clean_content:
        extracted = extract_all_tool_calls(clean_content)
        parsed_calls = []

        for json_obj in extracted:
            tool_name = (
                json_obj.get("name")
                or json_obj.get("tool")
                or json_obj.get("action")
            )
            tool_args = (
                json_obj.get("arguments")
                or json_obj.get("args")
                or json_obj.get("parameters")
                or json_obj.get("tool_input")
                or json_obj.get("action_input")
            )
            # If arguments are not nested under 'arguments' or 'args', collect all other keys
            if tool_args is None:
                tool_args = {
                    k: v
                    for k, v in json_obj.items()
                    if k not in ("name", "tool", "action", "type", "thought", "reasoning")
                }

            if isinstance(tool_args, dict):
                # Clean quotes from string values: e.g. "'models/endcap_tool.step'" -> "models/endcap_tool.step"
                tool_args = {
                    k: v.strip("'\"") if isinstance(v, str) else v
                    for k, v in tool_args.items()
                }

            if tool_name and isinstance(tool_name, str):
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        tool_args = {"input": tool_args}

                call_id = f"call_{uuid.uuid4().hex[:8]}"
                parsed_calls.append(
                    {
                        "name": tool_name.strip(),
                        "args": tool_args if isinstance(tool_args, dict) else {},
                        "id": call_id,
                        "type": "tool_call",
                    }
                )

        if not parsed_calls and messages:
            # Model generated conversational text before pipeline was finished
            fallback = determine_next_pipeline_call(messages)
            if fallback:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                parsed_calls.append(
                    {
                        "name": fallback["name"],
                        "args": fallback["arguments"],
                        "id": call_id,
                        "type": "tool_call",
                    }
                )
                TerminalLogger.info(
                    f"[Auto-Recovery] Reasoning model provided plan without tool command; "
                    f"advancing pipeline to active stage '{fallback['name']}'..."
                )

        if parsed_calls:
            # Enforce single tool execution per turn so subsequent steps observe real file paths
            message.tool_calls = [parsed_calls[0]]
            message.content = ""

    return message


def format_tool_instructions(tools: Sequence[Any], is_thinking: bool = False) -> str:
    """Formats concise tool signatures for fast Ollama prompt evaluation."""
    if is_thinking:
        header_lines = [
            "\n### Available Tools:",
            "Reason briefly in <think>...</think> (under 40 words, max 2 sentences).",
            "Immediately after </think>, output EXACTLY ONE JSON tool command:",
            '```json\n{"name": "<tool_name>", "arguments": {<args>}}\n```',
            "Tools available for invocation:",
        ]
    else:
        header_lines = [
            "\n### Available Tools:",
            "In each step, output EXACTLY ONE JSON tool command:",
            '```json\n{"name": "<tool_name>", "arguments": {<args>}}\n```',
            "Tools available for invocation:",
        ]

    lines = list(header_lines)
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = getattr(t, "description", "").strip().split("\n")[0][:80]
        args_keys = []
        if hasattr(t, "args") and isinstance(t.args, dict):
            args_keys = list(t.args.keys())
        elif hasattr(t, "args_schema") and t.args_schema:
            try:
                args_keys = list(t.args_schema.model_json_schema().get("properties", {}).keys())
            except Exception:
                pass
        lines.append(f"• {name}({', '.join(args_keys)}): {desc}")

    lines.append("Stop immediately after outputting one JSON command.")
    return "\n".join(lines)


class LocalLLMChatOllama(ChatOllama):
    """Subclasses ChatOllama to transparently intercept local model responses,
    optionally handling <think> tags if in thinking mode, isolating JSON tool-calls,
    showing live CLI animation, and retrying on transient errors up to 3 attempts.
    """

    bound_tool_defs: List[Any] = Field(default_factory=list)
    max_retries: int = Field(default=3)
    num_ctx: int = Field(default=2048)
    is_thinking_model: bool = Field(default=False)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Any]],
        **kwargs: Any,
    ) -> Any:
        return LocalLLMChatOllama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=getattr(self, "num_ctx", 2048),
            max_retries=getattr(self, "max_retries", 3),
            is_thinking_model=getattr(self, "is_thinking_model", False),
            bound_tool_defs=list(tools),
        )

    def _inject_tool_prompt(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        if not self.bound_tool_defs:
            return messages

        tool_prompt = format_tool_instructions(self.bound_tool_defs, is_thinking=self.is_thinking_model)
        new_messages = list(messages)

        for i, msg in enumerate(new_messages):
            if isinstance(msg, SystemMessage):
                new_messages[i] = SystemMessage(content=msg.content + "\n" + tool_prompt)
                return new_messages

        new_messages.insert(0, SystemMessage(content=tool_prompt))
        return new_messages

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        kwargs.pop("tools", None)
        if isinstance(input, list):
            input = self._inject_tool_prompt(input)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            attempt_label = f" (attempt {attempt}/{self.max_retries})" if attempt > 1 else ""
            with TerminalLogger.spinner(f"{self.model} reasoning & planning{attempt_label}..."):
                try:
                    res = super().invoke(input, config=config, **kwargs)
                    if isinstance(res, AIMessage):
                        res = sanitize_ai_message(
                            res,
                            messages=input if isinstance(input, list) else None,
                            is_thinking_model=self.is_thinking_model,
                        )
                    return res
                except Exception as err:
                    last_err = err
                    if attempt < self.max_retries:
                        TerminalLogger.warn(f"LLM call attempt {attempt}/{self.max_retries} failed: {err}. Retrying in 1.5s...")
                        time.sleep(1.5)
                    else:
                        TerminalLogger.error(f"LLM call failed after {self.max_retries} attempts: {err}")
                        raise last_err

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        kwargs.pop("tools", None)
        if isinstance(input, list):
            input = self._inject_tool_prompt(input)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            attempt_label = f" (attempt {attempt}/{self.max_retries})" if attempt > 1 else ""
            with TerminalLogger.spinner(f"{self.model} reasoning & planning{attempt_label}..."):
                try:
                    res = await super().ainvoke(input, config=config, **kwargs)
                    if isinstance(res, AIMessage):
                        res = sanitize_ai_message(
                            res,
                            messages=input if isinstance(input, list) else None,
                            is_thinking_model=self.is_thinking_model,
                        )
                    return res
                except Exception as err:
                    last_err = err
                    if attempt < self.max_retries:
                        TerminalLogger.warn(f"LLM call attempt {attempt}/{self.max_retries} failed: {err}. Retrying in 1.5s...")
                        await asyncio.sleep(1.5)
                    else:
                        TerminalLogger.error(f"LLM call failed after {self.max_retries} attempts: {err}")
                        raise last_err

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("tools", None)
        messages = self._inject_tool_prompt(messages)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            attempt_label = f" (attempt {attempt}/{self.max_retries})" if attempt > 1 else ""
            with TerminalLogger.spinner(f"{self.model} reasoning & planning{attempt_label}..."):
                try:
                    result: ChatResult = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    for gen in result.generations:
                        if isinstance(gen.message, AIMessage):
                            gen.message = sanitize_ai_message(
                                gen.message,
                                messages=messages,
                                is_thinking_model=self.is_thinking_model,
                            )
                    return result
                except Exception as err:
                    last_err = err
                    if attempt < self.max_retries:
                        TerminalLogger.warn(f"LLM generation attempt {attempt}/{self.max_retries} failed: {err}. Retrying in 1.5s...")
                        time.sleep(1.5)
                    else:
                        TerminalLogger.error(f"LLM generation failed after {self.max_retries} attempts: {err}")
                        raise last_err

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs.pop("tools", None)
        messages = self._inject_tool_prompt(messages)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            attempt_label = f" (attempt {attempt}/{self.max_retries})" if attempt > 1 else ""
            with TerminalLogger.spinner(f"{self.model} reasoning & planning{attempt_label}..."):
                try:
                    result: ChatResult = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    for gen in result.generations:
                        if isinstance(gen.message, AIMessage):
                            gen.message = sanitize_ai_message(
                                gen.message,
                                messages=messages,
                                is_thinking_model=self.is_thinking_model,
                            )
                    return result
                except Exception as err:
                    last_err = err
                    if attempt < self.max_retries:
                        TerminalLogger.warn(f"LLM generation attempt {attempt}/{self.max_retries} failed: {err}. Retrying in 1.5s...")
                        await asyncio.sleep(1.5)
                    else:
                        TerminalLogger.error(f"LLM generation failed after {self.max_retries} attempts: {err}")
                        raise last_err


# Aliases for backwards compatibility
DeepSeekR1ChatOllama = LocalLLMChatOllama
DeepSeekR1ChatWrapper = LocalLLMChatOllama
