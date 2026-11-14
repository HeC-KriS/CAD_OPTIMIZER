"""CLI entrypoint for Autonomous Manufacturing Agent."""

import argparse
import asyncio
import os
import sys
from src.config import AgentSettings, LLMConfig
from src.prompts import DEFAULT_PIPELINE_QUERY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Manufacturing Agent: Ingest -> [nTop <-> FEA] -> Auto-Orient -> Slicer."
    )
    parser.add_argument(
        "--file_path",
        "--file-path",
        "--file",
        type=str,
        default="models/sample_bracket.step",
        help="Path to the target CAD model file (.step, .dxf, .stl). Defaults to 'models/sample_bracket.step'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional Ollama model override (defaults strictly to OLLAMA_MODEL from .env)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Optional Ollama API base URL override (defaults to OLLAMA_BASE_URL from .env or http://localhost:11434)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retry attempts for LLM generation and tool executions (limit 2-3, defaults to 3)",
    )
    parser.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        default=None,
        help="Explicitly enable thinking/reasoning model middleware (<think> tags). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Explicitly disable thinking model middleware (normal mode for Qwen Coder, Llama, etc.).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom pipeline instruction for the agent. If omitted, uses default closed-loop query for --file_path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from src.agent import run_agent
    except ModuleNotFoundError as e:
        print(
            f"[!] Missing required dependency: {e.name}.\n"
            f"[!] Please install dependencies first using:\n\n"
            f"    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model
    if args.ollama_url:
        llm_kwargs["base_url"] = args.ollama_url

    if args.retries:
        llm_kwargs["max_retries"] = min(max(args.retries, 1), 5)
    if args.thinking is not None:
        llm_kwargs["is_thinking_model"] = args.thinking

    try:
        settings = AgentSettings(
            llm=LLMConfig(**llm_kwargs),
        )
    except ValueError as val_err:
        print(f"[!] Configuration error: {val_err}", file=sys.stderr)
        sys.exit(1)

    # Determine execution query based on --file_path
    if args.query:
        query = args.query
    else:
        file_target = args.file_path
        query = (
            f"Execute the closed-loop manufacturing pipeline on '{file_target}'. "
            f"Do not just explain the steps: actively invoke the tools starting with 'load_cad_model'."
        )

    try:
        asyncio.run(run_agent(settings, query))
    except KeyboardInterrupt:
        print("\n[!] Execution cancelled by user.")
        sys.exit(0)
    except Exception as err:
        print(f"\n[!] Failed to execute agent: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
