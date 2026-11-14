# Autonomous Generative Manufacturing Agent (`ntop_optim`)

A local-first LangChain engineering agent powered by local Ollama models (**DeepSeek-R1**, **Qwen2.5-Coder**, etc.) that automates the closed-loop generative design and additive manufacturing pipeline:
**CAD Ingest** $\rightarrow$ **[nTop $\leftrightarrow$ FEA Closed-Loop]** $\rightarrow$ **Trimesh Auto-Orientation** $\rightarrow$ **High-Precision Slicing & G-Code Generation**.

---

## Directory Organization

All input models, converted intermediate meshes, oriented STLs, and final `.gcode` outputs live strictly in the [`models/`](file:///c:/Users/Nitesh%20Kumar%20S/Desktop/ntop_optim/models) directory:

```text
ntop_optim/
├── models/                     # Central repository for CAD, STL, and G-code
│   ├── endcap_tool.step        # Sample CAD tool design space
│   ├── sample_bracket.step     # Default input aerospace bracket design space
│   ├── sample_bracket.dxf      # 2D/3D DXF CAD drawing
│   ├── test_primitive.stl      # Generated primitive test mesh
│   ├── test_primitive_oriented.stl # Auto-oriented mesh
│   └── test_primitive.gcode    # Final generated machine-ready G-code
├── src/
│   ├── __init__.py             # Package exports
│   ├── config.py               # Reads OLLAMA_MODEL (supports any Ollama model) from .env
│   ├── logger.py               # Real-time formatted terminal logger with spinner animation
│   ├── prompts.py              # Dynamic closed-loop prompts adapting to thinking/standard models
│   ├── think_middleware.py     # Reasoning model <think> stripper & JSON payload isolator
│   ├── agent.py                # LangChain / LangGraph ReAct agent with automatic retries
│   └── tools/
│       ├── __init__.py         # Manufacturing tools registry
│       ├── cad.py              # Stage 1: Ingest CAD model (auto-resolves models/)
│       ├── ntop.py             # Stage 2/3: nTop topology optimization (closed-loop iteration support)
│       ├── fea_pinn.py         # Stage 2/3: FEA structural validation (feedback loop driver)
│       ├── orientation.py      # Stage 4: Trimesh convex hull auto-orientation (in models/)
│       └── slicer.py           # Stage 5: Headless PrusaSlicer & Built-in Geometric Slicer engine
├── tests/
│   ├── test_manufacturing.py   # Test suite supporting --file_path flag
│   └── test_middleware.py      # Unit tests for <think> stripping and JSON isolation
├── main.py                     # CLI entrypoint supporting --file_path flag
├── local_mcp_agent.py          # Backward-compatibility runner supporting --file_path
├── .env.example                # Example environment variables template
├── requirements.txt            # Dependencies (langchain, trimesh, shapely, networkx, pydantic)
└── README.md                   # Documentation
```

---

## Closed-Loop Optimization Pipeline

1. **CAD Ingest**: Loads `.step`, `.dxf`, or `.stl` geometry from `models/`.
2. **Iterative Optimization $\leftrightarrow$ FEA Validation (Loop Limit: 3)**:
   - **nTop**: Executes topology lightweighting for target volume fraction.
   - **FEA Analysis**: Evaluates structural Factor of Safety ($FoS$).
   - **Loop Decision**:
     - If $FoS < 1.5$ (`NOT GOOD`) and iteration $< 3$: Re-runs nTop with increased volume fraction.
     - If $FoS \ge 1.5$ (`GOOD`) or loop limit reached: Breaks loop and proceeds.
3. **Auto-Orientation**: Uses Trimesh convex hull resting analysis to seat the largest flat facet on the XY plane, minimizing Z-height and overhangs. Exports `models/<name>_oriented.stl`.
4. **Slicing & G-Code Generation**: Invokes headless `prusa-slicer` if available on the system PATH, or runs the built-in high-fidelity geometric slicing engine (`trimesh` + `shapely`) to calculate layer perimeters, rectilinear infill, and volumetric extrusion, exporting production-ready `models/<name>.gcode`.

---

## Usage

### Run Script (`main.py`)

```bash
# Run with default CAD model (models/sample_bracket.step)
python main.py

# Run with custom CAD model via --file_path
python main.py --file_path models/endcap_tool.step
python main.py --file_path models/sample_bracket.step

# Override Ollama model or URL if needed
python main.py --file_path models/endcap_tool.step --model qwen2.5-coder:3b
python main.py --file_path models/endcap_tool.step --model deepseek-r1:7b --thinking
```

### Test Suite (`tests/test_manufacturing.py`)

```bash
# Run test suite with default CAD model
python tests/test_manufacturing.py

# Run test suite on a custom CAD model
python tests/test_manufacturing.py --file_path models/endcap_tool.step
```
