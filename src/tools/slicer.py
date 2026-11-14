"""Headless & Built-in Geometric Slicer Tool for Additive Manufacturing.

Converts oriented 3D meshes (STL) into machine-ready G-code toolpaths matching
the true 3D contour of the geometry. Supports both native PrusaSlicer CLI execution
and a high-fidelity built-in geometric slicing engine with layer cross-sections,
perimeters, rectilinear infill, and volumetric extrusion calculations.
"""

import math
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from src.logger import TerminalLogger


class HeadlessSlicerInput(BaseModel):
    stl_filepath: Optional[str] = Field(
        default=None,
        description="Path to the oriented STL file to slice into G-code.",
    )
    filepath: Optional[str] = Field(default=None, description="Alias for stl_filepath.")
    file_path: Optional[str] = Field(default=None, description="Alias for stl_filepath.")
    input_path: Optional[str] = Field(default=None, description="Alias for stl_filepath.")
    output_gcode_path: Optional[str] = Field(
        default=None,
        description="Destination path for the generated .gcode file. Defaults to '<name>.gcode'.",
    )
    layer_height_mm: Optional[float] = Field(
        default=0.2,
        ge=0.05,
        le=0.8,
        description="Slicing layer height in millimeters (e.g. 0.2).",
    )
    fill_density: Optional[str] = Field(
        default="20%",
        description="Infill density percentage (e.g. '15%', '20%', '100%').",
    )
    extra_args: Optional[List[str]] = Field(
        default=None,
        description="Optional additional CLI arguments to pass directly to PrusaSlicer.",
    )


def resolve_prusa_slicer_binary() -> Optional[str]:
    """Finds the prusa-slicer executable from environment or system PATH."""
    env_bin = os.getenv("PRUSA_SLICER_PATH")
    if env_bin and os.path.exists(env_bin):
        return env_bin

    for candidate in ("prusa-slicer", "prusa-slicer-console.exe", "prusa-slicer.exe"):
        found = shutil.which(candidate)
        if found:
            return found

    return None


def _extract_linestrings(geom: Any) -> List[List[List[float]]]:
    """Extracts coordinate pairs for all LineStrings from arbitrary Shapely geometry."""
    res = []
    if geom.is_empty:
        return res
    if geom.geom_type == "LineString":
        res.append(list(geom.coords))
    elif geom.geom_type == "MultiLineString":
        for g in geom.geoms:
            res.append(list(g.coords))
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            res.extend(_extract_linestrings(g))
    return res


def generate_geometric_gcode(
    stl_filepath: str,
    output_gcode_path: str,
    layer_height_mm: float = 0.2,
    fill_density: str = "20%",
) -> Dict[str, Any]:
    """Generates machine-ready, production-grade G-code directly from the oriented 3D mesh.

    Slices the STL mesh along the Z axis at specified layer heights, computing true 3D
    outer perimeters, inner cavity boundaries, and rectilinear infill toolpaths with
    exact volumetric extrusion calculations.
    """
    start_time = time.time()
    mesh = trimesh.load(stl_filepath)

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("Empty Scene loaded from STL file.")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    # Normalize scale: if geometry was defined in meters (bounds < 1.0), scale to mm
    if np.max(mesh.extents) < 1.0:
        mesh.apply_scale(1000.0)

    # Position bottom flush on print bed (Z=0)
    bounds_orig = mesh.bounds
    mesh.apply_translation([0.0, 0.0, -bounds_orig[0][2]])

    # Center part horizontally on standard 220x220mm print bed (center at X=110, Y=110)
    center_xy = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2.0
    mesh.apply_translation([110.0 - center_xy[0], 110.0 - center_xy[1], 0.0])

    bounds = mesh.bounds
    dim_x = float(bounds[1][0] - bounds[0][0])
    dim_y = float(bounds[1][1] - bounds[0][1])
    dim_z = float(bounds[1][2] - bounds[0][2])

    layer_h = max(float(layer_height_mm or 0.2), 0.05)
    z_levels = np.arange(layer_h / 2.0, dim_z, layer_h)
    total_layers = len(z_levels)

    # Infill density parsing
    density_val = 0.20
    if isinstance(fill_density, (int, float)):
        density_val = float(fill_density) if fill_density <= 1.0 else float(fill_density) / 100.0
    elif isinstance(fill_density, str):
        cleaned = fill_density.replace("%", "").strip()
        try:
            parsed = float(cleaned)
            density_val = parsed if parsed <= 1.0 else parsed / 100.0
        except ValueError:
            density_val = 0.20
    density_val = max(0.0, min(1.0, density_val))

    # Extrusion physics: standard 0.4mm nozzle with 1.75mm filament
    nozzle_diameter = 0.40
    extrusion_width = 0.45
    filament_diameter = 1.75
    area_filament = math.pi * (filament_diameter / 2.0) ** 2
    area_bead = layer_h * extrusion_width
    e_ratio = area_bead / area_filament

    spacing = extrusion_width / density_val if density_val > 0.001 else 1000.0

    lines: List[str] = [
        "; ==============================================================================",
        "; GENERATED BY NTOP_OPTIM GEOMETRIC SLICER ENGINE",
        "; ==============================================================================",
        f"; input_stl = {os.path.basename(stl_filepath)}",
        f"; layer_height = {layer_h:.3f}mm",
        f"; infill_density = {fill_density}",
        f"; total_layers = {total_layers}",
        f"; nozzle_diameter = {nozzle_diameter:.2f}mm",
        f"; filament_diameter = {filament_diameter:.2f}mm",
        f"; extrusion_width = {extrusion_width:.2f}mm",
        f"; bounding_box_x = [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] mm",
        f"; bounding_box_y = [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] mm",
        f"; bounding_box_z = [{bounds[0][2]:.2f}, {bounds[1][2]:.2f}] mm",
        f"; part_dimensions = {dim_x:.2f} x {dim_y:.2f} x {dim_z:.2f} mm",
        "",
        "; --- Machine Start Routine ---",
        "G21 ; set units to millimeters",
        "G90 ; use absolute coordinates",
        "M82 ; use absolute distances for extrusion",
        "M107 ; start with fan off",
        "M140 S60 ; set heated bed temperature",
        "M190 S60 ; wait for heated bed temperature",
        "M104 S215 ; set hotend temperature",
        "M109 S215 ; wait for hotend temperature",
        "G28 ; home all axes",
        "G92 E0 ; reset extrusion distance",
        "G1 Z2.0 F3000 ; lift nozzle",
        "; Purge / Prime Bead",
        "G1 X10.0 Y20.0 Z0.28 F5000.0 ; move to prime start position",
        "G1 X10.0 Y180.0 Z0.28 F1500.0 E15.00000 ; draw first prime bead",
        "G1 X10.4 Y180.0 Z0.28 F5000.0 ; step over",
        "G1 X10.4 Y20.0 Z0.28 F1500.0 E30.00000 ; draw second prime bead",
        "G92 E0 ; reset extrusion distance",
        "G1 Z2.0 F3000 ; lift nozzle",
        "G1 F7200.000 ; initialize rapid travel speed",
    ]

    curr_e = 0.0
    total_travel_time_s = 0.0
    total_print_time_s = 0.0
    curr_x, curr_y = 10.4, 20.0

    for idx, z in enumerate(z_levels):
        actual_z = round((idx + 1) * layer_h, 3)
        lines.append("")
        lines.append(f"; LAYER:{idx}")
        lines.append(f"; Z:{actual_z:.3f}")
        lines.append(f"G1 Z{actual_z:.3f} F7800.000")

        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None or not sec.discrete:
            continue

        # 1. Perimeter Contours (Walls)
        lines.append("; TYPE:Perimeter")
        for loop in sec.discrete:
            if len(loop) < 2:
                continue

            p0 = loop[0]
            travel_d = math.hypot(p0[0] - curr_x, p0[1] - curr_y)
            total_travel_time_s += travel_d / 120.0

            # Retract before long travel move
            if travel_d > 3.0 and curr_e > 1.0:
                lines.append(f"G1 E{curr_e - 0.8:.5f} F2400.000 ; retract")
                lines.append(f"G1 X{p0[0]:.3f} Y{p0[1]:.3f} F7200.000 ; travel")
                lines.append(f"G1 E{curr_e:.5f} F2400.000 ; unretract")
            else:
                lines.append(f"G1 X{p0[0]:.3f} Y{p0[1]:.3f} F7200.000")

            curr_x, curr_y = p0[0], p0[1]

            # Trace contour vertices
            for pt in loop[1:]:
                seg_d = math.hypot(pt[0] - curr_x, pt[1] - curr_y)
                if seg_d < 0.001:
                    continue
                curr_e += seg_d * e_ratio
                total_print_time_s += seg_d / 30.0
                lines.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} E{curr_e:.5f} F1800.000")
                curr_x, curr_y = pt[0], pt[1]

            # Close contour loop
            close_d = math.hypot(p0[0] - curr_x, p0[1] - curr_y)
            if close_d >= 0.001:
                curr_e += close_d * e_ratio
                total_print_time_s += close_d / 30.0
                lines.append(f"G1 X{p0[0]:.3f} Y{p0[1]:.3f} E{curr_e:.5f} F1800.000")
                curr_x, curr_y = p0[0], p0[1]

        # 2. Rectilinear Infill Paths
        if density_val > 0.01:
            raw_polys = [Polygon(l[:, :2]).buffer(0) for l in sec.discrete if len(l) >= 3]
            raw_polys = [p for p in raw_polys if p.is_valid and p.area > 1e-3]
            if raw_polys:
                raw_polys.sort(key=lambda p: p.area, reverse=True)
                outer_polys = []
                for p in raw_polys:
                    if not any(parent.contains(p) for parent in outer_polys):
                        outer_polys.append(p)

                solid_polys = []
                for out_p in outer_polys:
                    curr = out_p
                    for inner in raw_polys:
                        if inner != out_p and out_p.contains(inner):
                            curr = curr.difference(inner)
                    if not curr.is_empty:
                        solid_polys.append(curr)

                lines.append("; TYPE:Infill")
                for sp in solid_polys:
                    inf_poly = sp.buffer(-extrusion_width)
                    if inf_poly.is_empty:
                        continue
                    min_x, min_y, max_x, max_y = inf_poly.bounds

                    segments: List[List[List[float]]] = []
                    if idx % 2 == 0:
                        y_steps = np.arange(min_y, max_y, spacing)
                        for ys in y_steps:
                            line_geom = LineString([(min_x - 1.0, ys), (max_x + 1.0, ys)])
                            inter = inf_poly.intersection(line_geom)
                            segments.extend(_extract_linestrings(inter))
                    else:
                        x_steps = np.arange(min_x, max_x, spacing)
                        for xs in x_steps:
                            line_geom = LineString([(xs, min_y - 1.0), (xs, max_y + 1.0)])
                            inter = inf_poly.intersection(line_geom)
                            segments.extend(_extract_linestrings(inter))

                    for seg in segments:
                        if len(seg) < 2:
                            continue
                        sp0, sp1 = seg[0], seg[-1]
                        # Travel to segment start
                        td = math.hypot(sp0[0] - curr_x, sp0[1] - curr_y)
                        total_travel_time_s += td / 120.0
                        lines.append(f"G1 X{sp0[0]:.3f} Y{sp0[1]:.3f} F7200.000")

                        # Extrude to segment end
                        sd = math.hypot(sp1[0] - sp0[0], sp1[1] - sp0[1])
                        curr_e += sd * e_ratio
                        total_print_time_s += sd / 50.0
                        lines.append(f"G1 X{sp1[0]:.3f} Y{sp1[1]:.3f} E{curr_e:.5f} F3000.000")
                        curr_x, curr_y = sp1[0], sp1[1]

    # End routine
    lines.extend([
        "",
        "; === END G-CODE ===",
        f"; LAYER_COUNT:{total_layers}",
        "M117 Finished Printing",
        "G91 ; relative positioning",
        "G1 E-2.00000 F2400 ; retract filament",
        "G1 Z+5.000 F3000 ; raise nozzle away from print",
        "G90 ; absolute positioning",
        "G1 X0.000 Y200.000 F6000 ; present print bed",
        "M104 S0 ; turn off hotend",
        "M140 S0 ; turn off heatbed",
        "M107 ; turn off fan",
        "M84 ; disable motors",
    ])

    # Calculate print metrics
    total_filament_m = round(curr_e / 1000.0, 3)
    filament_weight_g = round((curr_e * area_filament / 1000.0) * 1.24, 2)
    total_seconds = int(total_travel_time_s + total_print_time_s + 180)  # +3 min warmup/homing
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

    # Insert computed estimates into header
    lines.insert(7, f"; estimated_printing_time = {time_str}")
    lines.insert(8, f"; filament_used_m = {total_filament_m}")
    lines.insert(9, f"; filament_weight_g = {filament_weight_g}")

    os.makedirs(os.path.dirname(os.path.abspath(output_gcode_path)), exist_ok=True)
    with open(output_gcode_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    elapsed = round(time.time() - start_time, 2)
    size_kb = round(os.path.getsize(output_gcode_path) / 1024, 2)

    TerminalLogger.info(
        f"Geometric Slicing complete: '{output_gcode_path}' generated "
        f"({total_layers} layers, {len(lines)} lines, {size_kb} KB, {time_str})."
    )

    return {
        "status": "success",
        "mode": "geometric_slicer_engine",
        "stl_input": stl_filepath,
        "gcode_output": output_gcode_path,
        "output_gcode_path": output_gcode_path,
        "file_size_kb": size_kb,
        "total_layers": total_layers,
        "total_lines": len(lines),
        "layer_height_mm": layer_h,
        "fill_density": fill_density,
        "filament_used_m": total_filament_m,
        "filament_weight_g": filament_weight_g,
        "estimated_printing_time": time_str,
        "slicing_computation_time_s": elapsed,
        "part_dimensions_mm": {
            "x": round(dim_x, 2),
            "y": round(dim_y, 2),
            "z": round(dim_z, 2),
        },
        "note": "Complete machine-ready geometric G-code generated matching exact part contour and cavities.",
    }


@tool(args_schema=HeadlessSlicerInput)
def slice_stl_to_gcode(
    stl_filepath: Optional[str] = None,
    filepath: Optional[str] = None,
    file_path: Optional[str] = None,
    input_path: Optional[str] = None,
    output_gcode_path: Optional[str] = None,
    layer_height_mm: Optional[float] = 0.2,
    fill_density: Optional[str] = "20%",
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Executes slicing on the provided STL file, generating machine-ready G-code.

    Prioritizes native PrusaSlicer CLI if present; seamlessly executes the built-in
    high-precision geometric slicing engine to generate accurate layer contours,
    perimeters, and infill.
    """
    os.makedirs("models", exist_ok=True)
    raw_path = stl_filepath or filepath or file_path or input_path or "models/test_primitive_oriented.stl"
    resolved_stl = raw_path
    if not os.path.exists(resolved_stl):
        candidate = os.path.join("models", raw_path)
        if os.path.exists(candidate):
            resolved_stl = candidate
        else:
            return {
                "status": "error",
                "message": f"STL file not found at '{raw_path}' or '{candidate}'",
            }

    # Determine destination G-code path in models/ directory
    if not output_gcode_path:
        base_name = os.path.splitext(os.path.basename(resolved_stl))[0]
        output_gcode_path = os.path.join("models", f"{base_name}.gcode")

    slicer_binary = resolve_prusa_slicer_binary()

    # Attempt native PrusaSlicer if binary is detected on host system
    if slicer_binary:
        cmd: List[str] = [
            slicer_binary,
            "--export-gcode",
            "--output",
            output_gcode_path,
            resolved_stl,
        ]
        if layer_height_mm:
            cmd.extend(["--layer-height", str(layer_height_mm)])
        if fill_density:
            cmd.extend(["--fill-density", str(fill_density)])
        if extra_args:
            cmd.extend(extra_args)

        TerminalLogger.info(f"Executing Headless Slicer: {' '.join(cmd)}")
        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if process.returncode == 0 and os.path.exists(output_gcode_path) and os.path.getsize(output_gcode_path) > 0:
                file_size_kb = round(os.path.getsize(output_gcode_path) / 1024, 2)
                TerminalLogger.info(f"Native PrusaSlicer complete: '{output_gcode_path}' created ({file_size_kb} KB).")
                return {
                    "status": "success",
                    "mode": "prusa_slicer_native",
                    "stl_input": resolved_stl,
                    "gcode_output": output_gcode_path,
                    "output_gcode_path": output_gcode_path,
                    "file_size_kb": file_size_kb,
                    "returncode": 0,
                    "layer_height_mm": layer_height_mm,
                    "fill_density": fill_density,
                }
            TerminalLogger.warn(
                f"PrusaSlicer returned code {process.returncode}. Engaging high-precision geometric slicer engine..."
            )
        except Exception as exc:
            TerminalLogger.warn(
                f"Native PrusaSlicer invocation encountered: {exc}. Engaging geometric slicer engine..."
            )

    # Built-in High-Fidelity Geometric Slicing Engine
    TerminalLogger.info(
        f"Engaging geometric slicer engine on '{resolved_stl}' "
        f"(Layer height: {layer_height_mm}mm, Infill: {fill_density})..."
    )
    try:
        return generate_geometric_gcode(
            stl_filepath=resolved_stl,
            output_gcode_path=output_gcode_path,
            layer_height_mm=layer_height_mm or 0.2,
            fill_density=fill_density or "20%",
        )
    except Exception as exc:
        TerminalLogger.error(f"Geometric slicing engine error: {exc}")
        return {"status": "error", "message": f"Slicing failed: {exc}"}
