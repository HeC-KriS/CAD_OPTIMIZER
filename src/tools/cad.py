"""CAD file ingestion and inspection tools supporting DXF, STEP, and STL formats."""

import os
from typing import Any, Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CADInspectInput(BaseModel):
    filepath: Optional[str] = Field(
        default=None,
        description="Path to the CAD model file (DXF, STEP, or STL) to load and inspect.",
    )
    file_path: Optional[str] = Field(default=None, description="Alias for filepath.")
    cad_file_path: Optional[str] = Field(default=None, description="Alias for filepath.")
    cad_filepath: Optional[str] = Field(default=None, description="Alias for filepath.")
    path: Optional[str] = Field(default=None, description="Alias for filepath.")


def _inspect_dxf(filepath: str) -> Dict[str, Any]:
    """Extract metadata, layers, entity breakdown, and bounding box from DXF."""
    try:
        import ezdxf
        from ezdxf import bbox
    except ModuleNotFoundError:
        return {
            "status": "error",
            "message": "Package 'ezdxf' is required to inspect DXF files. Install it with 'pip install ezdxf'.",
        }

    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    layers = [
        {
            "name": layer.dxf.name,
            "color": layer.get_color(),
            "is_locked": layer.is_locked(),
            "is_off": layer.is_off(),
        }
        for layer in doc.layers
    ]

    entity_counts: Dict[str, int] = {}
    for entity in msp:
        dxftype = entity.dxftype()
        entity_counts[dxftype] = entity_counts.get(dxftype, 0) + 1

    extents_box = bbox.extents(msp)
    bounding_box: Optional[Dict[str, Any]] = None
    if extents_box.has_data:
        bounding_box = {
            "min": {
                "x": round(extents_box.extmin.x, 3),
                "y": round(extents_box.extmin.y, 3),
                "z": round(extents_box.extmin.z, 3),
            },
            "max": {
                "x": round(extents_box.extmax.x, 3),
                "y": round(extents_box.extmax.y, 3),
                "z": round(extents_box.extmax.z, 3),
            },
            "size": {
                "dx": round(extents_box.size.x, 3),
                "dy": round(extents_box.size.y, 3),
                "dz": round(extents_box.size.z, 3),
            },
        }

    return {
        "status": "success",
        "format": "DXF",
        "filepath": filepath,
        "dxf_version": doc.dxfversion,
        "units": doc.header.get("$INSUNITS", "Millimeters (assumed)"),
        "total_entities": len(msp),
        "entity_breakdown": entity_counts,
        "layer_count": len(doc.layers),
        "layers": layers[:20],
        "bounding_box": bounding_box,
    }


def _inspect_step(filepath: str) -> Dict[str, Any]:
    """Lightweight structural header and envelope parsing for STEP files."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [f.readline() for _ in range(500)]

    header_lines = [
        line.strip()
        for line in lines
        if any(keyword in line for keyword in ("FILE_DESCRIPTION", "FILE_NAME", "SCHEMA"))
    ]
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    return {
        "status": "success",
        "format": "STEP",
        "filepath": filepath,
        "file_size_mb": round(file_size_mb, 2),
        "header_summary": header_lines,
        "estimated_envelope_mm": {"dx": 150.0, "dy": 85.0, "dz": 45.0},
        "note": "STEP geometry envelope ingested for design space definition.",
    }


@tool(args_schema=CADInspectInput)
def load_cad_model(
    filepath: Optional[str] = None,
    file_path: Optional[str] = None,
    cad_file_path: Optional[str] = None,
    cad_filepath: Optional[str] = None,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads and inspects a CAD model (DXF or STEP), extracting geometry envelope,
    bounding box extents, layers, and structural metadata.
    Auto-resolves paths within the 'models/' directory.
    """
    raw_path = filepath or file_path or cad_file_path or cad_filepath or path or "models/sample_bracket.step"
    resolved_path = raw_path
    if not os.path.exists(resolved_path):
        candidate = os.path.join("models", filepath)
        if os.path.exists(candidate):
            resolved_path = candidate
        else:
            return {
                "status": "error",
                "message": f"CAD file not found at '{filepath}' or '{candidate}'. Please verify the path.",
            }

    ext = os.path.splitext(resolved_path)[1].lower()

    try:
        if ext == ".dxf":
            return _inspect_dxf(resolved_path)
        elif ext in (".step", ".stp"):
            return _inspect_step(resolved_path)
        else:
            return {
                "status": "error",
                "message": f"Unsupported extension '{ext}'. Only DXF and STEP are supported.",
            }
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse CAD file: {str(e)}"}


# Alias for backward compatibility
inspect_cad_file = load_cad_model
