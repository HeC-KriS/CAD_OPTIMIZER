"""Backward-compatible entrypoint for the manufacturing agent."""

import argparse
import asyncio
import sys
from src.config import AgentSettings, LLMConfig
from src.prompts import DEFAULT_PIPELINE_QUERY


async def run_agent(user_query: str = DEFAULT_PIPELINE_QUERY):
    """Executes the agent using the modular src package."""
    try:
        from src.agent import run_agent as _modular_run_agent
    except ModuleNotFoundError as e:
        print(
            f"[!] Missing required dependency: {e.name}.\n"
            f"[!] Please install dependencies first using:\n\n"
            f"    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = AgentSettings(
        llm=LLMConfig(),
    )
    await _modular_run_agent(settings, user_query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manufacturing agent runner.")
    parser.add_argument(
        "--file_path",
        "--file-path",
        "--file",
        type=str,
        default="models/sample_bracket.step",
        help="Path to the target CAD model file (.step). Defaults to 'models/sample_bracket.step'",
    )
    args = parser.parse_args()

    query = (
        f"Execute the closed-loop manufacturing pipeline on '{args.file_path}'. "
        f"Do not just explain the steps: actively invoke the tools starting with 'load_cad_model'."
    )

    try:
        asyncio.run(run_agent(query))
    except KeyboardInterrupt:
        print("\n[!] Execution cancelled by user.")
    except Exception as err:
        print(f"\n[!] Failed to execute agent: {err}", file=sys.stderr)
