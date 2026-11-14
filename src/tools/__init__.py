"""Agent manufacturing tools and adapters."""

from .cad import CADInspectInput, inspect_cad_file, load_cad_model
from .fea_pinn import FEAPINNAnalysisInput, run_fea_pinn_analysis
from .ntop import MockNTopInput, run_ntop_topology_optimization
from .orientation import AutoOrientationInput, auto_orient_stl, compute_optimal_orientation
from .slicer import HeadlessSlicerInput, slice_stl_to_gcode

__all__ = [
    "load_cad_model",
    "inspect_cad_file",
    "CADInspectInput",
    "run_ntop_topology_optimization",
    "MockNTopInput",
    "run_fea_pinn_analysis",
    "FEAPINNAnalysisInput",
    "auto_orient_stl",
    "AutoOrientationInput",
    "compute_optimal_orientation",
    "slice_stl_to_gcode",
    "HeadlessSlicerInput",
]
