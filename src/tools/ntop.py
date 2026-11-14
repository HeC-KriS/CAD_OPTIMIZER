"""Mocked nTop Topology Optimization Tool with Iterative Feedback Support.

Executes within the closed-loop optimization cycle (nTop -> FEA -> nTop),
storing intermediate models in the 'models/' directory.
"""

import os
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.logger import TerminalLogger


class MockNTopInput(BaseModel):
    cad_filepath: Optional[str] = Field(
        default=None,
        description="Path to the CAD file (STEP, STL, etc.) to optimize.",
    )
    filepath: Optional[str] = Field(default=None, description="Alias for cad_filepath.")
    file_path: Optional[str] = Field(default=None, description="Alias for cad_filepath.")
    target_volume_fraction: Optional[float] = Field(
        default=0.3,
        description="Target volume fraction remaining (e.g. 0.3 for 70% reduction).",
    )
    iteration: Optional[int] = Field(
        default=1,
        ge=1,
        description="Current optimization loop iteration counter.",
    )
    max_iterations: Optional[int] = Field(
        default=3,
        description="Maximum loop limit before exiting to orientation and slicing.",
    )
    material: Optional[str] = Field(
        default="Ti-6Al-4V",
        description="Target alloy for density and compliance calculations.",
    )


@tool(args_schema=MockNTopInput)
def run_ntop_topology_optimization(
    cad_filepath: Optional[str] = None,
    filepath: Optional[str] = None,
    file_path: Optional[str] = None,
    target_volume_fraction: Optional[float] = 0.3,
    iteration: Optional[int] = 1,
    max_iterations: Optional[int] = 3,
    material: Optional[str] = "Ti-6Al-4V",
) -> Dict[str, Any]:
    """Executes an iteration of nTop topology optimization.
    Outputs optimized model into 'models/' and reports parameter metrics for FEA
    evaluation.
    """
    os.makedirs("models", exist_ok=True)
    raw_path = cad_filepath or filepath or file_path or "models/sample_bracket.step"
    resolved_path = raw_path
    if not os.path.exists(resolved_path):
        candidate = os.path.join("models", cad_filepath)
        if os.path.exists(candidate):
            resolved_path = candidate

    base_name = os.path.splitext(os.path.basename(resolved_path))[0]
    out_file = os.path.join("models", f"{base_name}_ntop_iter{iteration}.step")

    iter_num = iteration or 1
    max_iter = max_iterations or 3
    vol_frac = target_volume_fraction or 0.3

    TerminalLogger.info(
        f"[Optimization Loop] Executing nTop Iteration {iter_num}/{max_iter} "
        f"(Target Volume Fraction: {vol_frac}, Material: {material})"
    )
    TerminalLogger.info("nTop optimization step skipped/mocked (mock mode)")

    return {
        "status": "success",
        "stage": "nTop Topology Optimization (Mocked)",
        "iteration": iter_num,
        "max_iterations": max_iter,
        "target_volume_fraction": vol_frac,
        "material": material,
        "input_file": resolved_path,
        "output_file": resolved_path,  # Passed for FEA validation
        "next_step": "FEA_STRUCTURAL_VALIDATION",
        "message": (
            f"nTop iteration {iter_num}/{max_iter} completed. "
            f"Model ready for Stage 2 FEA evaluation."
        ),
    }
