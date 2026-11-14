"""DeepSeek-R1 Autonomous Manufacturing Agent Pipeline.

Orchestrates:
1. CAD Ingestion & Inspection
2. FEA / PINN Structural Analysis (Placeholder)
3. Mocked nTop Topology Optimization Step
4. Trimesh Convex Hull Auto-Orientation (Converts STEP to STL)
5. Headless PrusaSlicer G-Code Generation
"""

import time
from typing import Any, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .config import AgentSettings, LLMConfig
from .logger import TerminalLogger
from .prompts import SYSTEM_PROMPT, get_system_prompt
from .think_middleware import DeepSeekR1ChatOllama, LocalLLMChatOllama, strip_think_tags
from .tools.cad import load_cad_model
from .tools.fea_pinn import run_fea_pinn_analysis
from .tools.ntop import run_ntop_topology_optimization
from .tools.orientation import auto_orient_stl
from .tools.slicer import slice_stl_to_gcode


def get_llm(config: LLMConfig) -> Any:
    """Initializes local ChatOllama LLM configured with dynamic thinking middleware flag."""
    is_thinking = config.resolve_is_thinking_model()
    return LocalLLMChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        max_retries=config.max_retries,
        is_thinking_model=is_thinking,
    )


def make_retryable_tool(tool: BaseTool, max_retries: int = 3) -> BaseTool:
    """Wraps a BaseTool with automatic retry execution and CLI spinner animation."""
    original_func = getattr(tool, "func", None)
    if not original_func:
        return tool

    def retry_wrapper(*args: Any, **kwargs: Any) -> Any:
        last_res = None
        for attempt in range(1, max_retries + 1):
            attempt_str = f" (attempt {attempt}/{max_retries})" if attempt > 1 else ""
            with TerminalLogger.spinner(f"Executing {tool.name}{attempt_str}..."):
                try:
                    res = original_func(*args, **kwargs)
                    if isinstance(res, dict) and res.get("status") == "error":
                        last_res = res
                        if attempt < max_retries:
                            TerminalLogger.warn(
                                f"Tool {tool.name} returned error: {res.get('message')}. "
                                f"Retrying attempt {attempt + 1}/{max_retries}..."
                            )
                            time.sleep(1.0)
                            continue
                    return res
                except Exception as exc:
                    last_res = {"status": "error", "message": str(exc)}
                    if attempt < max_retries:
                        TerminalLogger.warn(
                            f"Tool {tool.name} encountered exception: {exc}. "
                            f"Retrying attempt {attempt + 1}/{max_retries}..."
                        )
                        time.sleep(1.0)
                    else:
                        TerminalLogger.error(f"Tool {tool.name} failed after {max_retries} attempts.")
                        raise exc
        return last_res

    tool.func = retry_wrapper
    return tool


def get_manufacturing_tools(max_retries: int = 3) -> List[BaseTool]:
    """Returns the manufacturing pipeline tools wrapped with retry support."""
    raw_tools = [
        load_cad_model,                  # Stage 1: CAD Ingest (STEP/DXF)
        run_fea_pinn_analysis,           # Stage 2: FEA Analysis (Placeholder)
        run_ntop_topology_optimization,  # Stage 3: nTop Optimization (Mocked)
        auto_orient_stl,                 # Stage 4: CAD to STL & Auto-Orientation
        slice_stl_to_gcode,              # Stage 5: Headless Slicing -> G-Code
    ]
    return [make_retryable_tool(t, max_retries=max_retries) for t in raw_tools]


def create_manufacturing_agent(llm: Any, tools: List[BaseTool], system_prompt: str = SYSTEM_PROMPT) -> Any:
    """Builds a LangChain / LangGraph ReAct agent equipped with manufacturing tools."""
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )


async def run_agent(settings: AgentSettings, user_query: str) -> None:
    """Runs the manufacturing pipeline agent with continuous, formatted terminal logs."""
    is_thinking = settings.llm.resolve_is_thinking_model()
    model_name = settings.llm.model
    system_prompt = get_system_prompt(model_name=model_name, is_thinking=is_thinking)

    TerminalLogger.header(f"Autonomous Manufacturing Agent ({model_name})")
    TerminalLogger.info(f"LLM Endpoint : {settings.llm.base_url}")
    mode_str = "Thinking / Reasoning Model (<think> enabled)" if is_thinking else "Standard Model (Normal Mode)"
    TerminalLogger.info(f"Model Engine : {model_name} [Mode: {mode_str}]")

    llm = get_llm(settings.llm)
    tools = get_manufacturing_tools(max_retries=settings.llm.max_retries)

    TerminalLogger.info(f"Active Pipeline Tools ({len(tools)}):")
    for idx, t in enumerate(tools, 1):
        TerminalLogger.info(f"  [{idx}] {t.name:<30} -> {t.description[:60]}...")

    agent = create_manufacturing_agent(llm=llm, tools=tools, system_prompt=system_prompt)

    TerminalLogger.header(f"Query: {user_query}")
    TerminalLogger.info(f"Starting ReAct agent execution loop with {model_name}...")

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]
    }

    final_report = ""

    async for chunk in agent.astream(inputs, stream_mode="updates"):
        for _, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if is_thinking and isinstance(msg.content, str) and "<think>" in msg.content:
                        clean_text, thought_text = strip_think_tags(msg.content)
                        if thought_text:
                            TerminalLogger.thought(thought_text)
                        msg.content = clean_text

                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            TerminalLogger.tool_call(tc["name"], tc["args"])
                    elif msg.content:
                        final_report = msg.content
                elif isinstance(msg, ToolMessage):
                    TerminalLogger.tool_result(msg.name or "Tool", msg.content)

    if final_report:
        TerminalLogger.final_answer(final_report)
