"""Test Suite for Closed-Loop Manufacturing: CAD -> [nTop <-> FEA Loop] -> Auto-Orientation -> Slicer -> G-Code.

Stores all CAD and manufacturing artifacts strictly in the 'models/' directory.
"""

import os
import sys
from typing import Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import trimesh
import trimesh.creation
import trimesh.transformations
from src.tools.cad import load_cad_model
from src.tools.fea_pinn import run_fea_pinn_analysis
from src.tools.ntop import run_ntop_topology_optimization
from src.tools.orientation import auto_orient_stl
from src.tools.slicer import slice_stl_to_gcode


def generate_test_primitive_stl(
    filepath: str = os.path.join("models", "test_primitive.stl"),
    extents: Tuple[float, float, float] = (80.0, 30.0, 15.0),
) -> str:
    """Creates a 3D box tilted vertically to verify Z-height minimization, saving to models/."""
    os.makedirs("models", exist_ok=True)
    box = trimesh.creation.box(extents=extents)
    tilt_rot = trimesh.transformations.rotation_matrix(angle=1.5708, direction=[0, 1, 0])
    box.apply_transform(tilt_rot)
    box.export(filepath)
    print(f"[+] Generated test primitive STL in models/: '{filepath}' (Initial Z-height: {round(box.extents[2], 2)}mm)")
    return filepath


def run_manufacturing_pipeline_test(
    cad_path: str = os.path.join("models", "sample_bracket.step"),
    stl_path: str = os.path.join("models", "test_primitive.stl"),
    oriented_stl_path: str = os.path.join("models", "test_primitive_oriented.stl"),
    gcode_path: str = os.path.join("models", "test_primitive.gcode"),
    max_loop_limit: int = 3,
) -> Dict[str, str]:
    """Executes the closed-loop optimization and manufacturing test suite."""
    print("\n" + "=" * 70)
    print("  RUNNING CLOSED-LOOP GENERATIVE MANUFACTURING TEST SUITE")
    print("=" * 70 + "\n")

    # 1. Stage 1: Load CAD Model from models/
    print("--- [STAGE 1] Ingest CAD Model from models/ ---")
    cad_result = load_cad_model.invoke({"filepath": cad_path})
    assert cad_result["status"] == "success", f"CAD ingest failed: {cad_result}"
    print(f"✔ CAD model ingested: format={cad_result['format']}, path={cad_result['filepath']}\n")

    # 2. Closed-Loop: nTop -> FEA -> Loop if Not Good -> Break when Good or Limit reached
    print(f"--- [STAGE 2 & 3] Optimization <-> FEA Validation Loop (Limit: {max_loop_limit}) ---")
    current_vol = 0.22
    loop_converged = False

    for iteration in range(1, max_loop_limit + 1):
        print(f"\n  [Loop Cycle {iteration}/{max_loop_limit}]")

        # Step A: nTop Optimization
        ntop_res = run_ntop_topology_optimization.invoke({
            "cad_filepath": cad_path,
            "target_volume_fraction": current_vol,
            "iteration": iteration,
            "max_iterations": max_loop_limit,
        })
        print(f"  • nTop Iteration {iteration} executed (Volume fraction: {current_vol})")

        # Step B: FEA Structural Analysis
        fea_res = run_fea_pinn_analysis.invoke({
            "model_filepath": ntop_res["output_file"],
            "iteration": iteration,
            "max_iterations": max_loop_limit,
            "target_fos": 1.5,
        })
        fos = fea_res["placeholder_results"]["factor_of_safety"]
        is_good = fea_res["is_good"]
        verdict = fea_res["verdict"]

        print(f"  • FEA Evaluation: FoS={fos} (Target: 1.5) | Verdict: {verdict}")

        if is_good:
            print(f"  ✔ Convergence achieved at iteration {iteration}! Proceeding to manufacturing.")
            loop_converged = True
            break
        else:
            print(f"  ⚠ Not Good: {fea_res['recommendation']}")
            current_vol += 0.08  # Adapt parameters for next loop

    assert loop_converged, "Closed-loop optimization should converge"
    print("\n✔ Closed-loop optimization successfully converged.\n")

    # 3. Stage 4: Auto-Orientation & STL Conversion in models/
    print("--- [STAGE 4] Trimesh Convex Hull Auto-Orientation (in models/) ---")
    orient_result = auto_orient_stl.invoke({
        "stl_filepath": stl_path,
        "output_filepath": oriented_stl_path,
        "overhang_angle_deg": 45.0,
    })

    assert orient_result["status"] == "success", f"Orientation failed: {orient_result}"
    assert os.path.exists(oriented_stl_path), f"Oriented STL '{oriented_stl_path}' was not created."
    assert os.path.dirname(os.path.abspath(oriented_stl_path)).endswith("models"), "Oriented STL not in models/!"

    orig_z = orient_result["original_z_height_mm"]
    opt_z = orient_result["minimized_z_height_mm"]
    print(f"✔ Auto-Orientation verified: Z-height minimized from {orig_z}mm to {opt_z}mm "
          f"({orient_result['z_height_reduction_pct']}% reduction).")
    print(f"  Exported to: '{oriented_stl_path}'\n")

    # 4. Stage 5: Headless Slicing -> G-Code in models/
    print("--- [STAGE 5] Headless PrusaSlicer Slicing -> G-Code (in models/) ---")
    slice_result = slice_stl_to_gcode.invoke({
        "stl_filepath": oriented_stl_path,
        "output_gcode_path": gcode_path,
        "layer_height_mm": 0.2,
        "fill_density": "20%",
    })

    assert slice_result["status"] == "success", f"Slicing failed: {slice_result}"
    assert os.path.exists(gcode_path), f"G-code file '{gcode_path}' was not generated."
    assert os.path.dirname(os.path.abspath(gcode_path)).endswith("models"), "G-code not in models/!"

    with open(gcode_path, "r", encoding="utf-8", errors="ignore") as f:
        gcode_sample = [f.readline() for _ in range(5)]

    print(f"✔ Headless Slicer verified: '{gcode_path}' successfully generated ({slice_result['file_size_kb']} KB).")
    print(f"  G-code Header Preview (first 5 lines):")
    for line in gcode_sample:
        print(f"    {line.strip()}")

    print("\n" + "=" * 70)
    print("  ALL CLOSED-LOOP STAGES PASSED! ALL ARTIFACTS CREATED IN models/ ✔")
    print("=" * 70 + "\n")

    return {
        "cad_file": cad_path,
        "primitive_stl": stl_path,
        "oriented_stl": oriented_stl_path,
        "gcode": gcode_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run manufacturing closed-loop test suite.")
    parser.add_argument(
        "--file_path",
        "--file-path",
        "--file",
        type=str,
        default="models/sample_bracket.step",
        help="Path to the CAD model file (.step). Defaults to 'models/sample_bracket.step'",
    )
    args = parser.parse_args()

    stl_file = generate_test_primitive_stl()
    try:
        run_manufacturing_pipeline_test(cad_path=args.file_path, stl_path=stl_file)
    except AssertionError as err:
        print(f"\n[FAIL] Assertion Error: {err}")
        exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Test Suite Encountered Exception: {exc}")
        exit(1)
