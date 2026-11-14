"""Autonomous Manufacturing Agent with Trimesh Auto-Orientation and Headless Slicing."""

from .config import AgentSettings, LLMConfig
from .prompts import DEFAULT_PIPELINE_QUERY, ENGINEERING_AGENT_SYSTEM_PROMPT, SYSTEM_PROMPT
from .tools.cad import inspect_cad_file, load_cad_model
from .tools.ntop import run_ntop_topology_optimization
from .tools.orientation import auto_orient_stl, compute_optimal_orientation
from .tools.slicer import slice_stl_to_gcode

__all__ = [
    "AgentSettings",
    "LLMConfig",
    "load_cad_model",
    "inspect_cad_file",
    "run_ntop_topology_optimization",
    "auto_orient_stl",
    "compute_optimal_orientation",
    "slice_stl_to_gcode",
    "create_agent",
    "get_llm",
    "run_agent",
    "SYSTEM_PROMPT",
    "ENGINEERING_AGENT_SYSTEM_PROMPT",
    "DEFAULT_PIPELINE_QUERY",
]


def __getattr__(name: str):
    if name in ("create_agent", "get_llm", "run_agent"):
        from . import agent as agent_module

        return getattr(agent_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
