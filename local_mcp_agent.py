"""
Local-First Autonomous Agent with MCP (ntop) and Local CAD Inspection.
Runs with LangGraph and Ollama (e.g., qwen2.5:14b or llama3.1:8b).
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, create_model

# LangChain & LangGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Model Context Protocol (MCP) SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
import mcp.types as mcp_types

# CAD parsing
import ezdxf
from ezdxf import bbox


# ============================================================================
# 1. Local CAD Inspection Tool
# ============================================================================

class CADInspectInput(BaseModel):
    filepath: str = Field(
        ...,
        description="Absolute or relative path to the CAD file (DXF or STEP format) to inspect.",
    )


@tool(args_schema=CADInspectInput)
def inspect_cad_file(filepath: str) -> Dict[str, Any]:
    """Inspects a local CAD drawing (DXF) to extract metadata, layers, entity

    counts, and 3D/2D bounding box geometry.
    """
    if not os.path.exists(filepath):
        return {"error": f"File not found at path: {filepath}"}

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".dxf":
        try:
            doc = ezdxf.readfile(filepath)
            msp = doc.modelspace()

            # Layer extraction
            layers = [
                {
                    "name": layer.dxf.name,
                    "color": layer.get_color(),
                    "is_locked": layer.is_locked(),
                    "is_off": layer.is_off(),
                }
                for layer in doc.layers
            ]

            # Entity counts
            entity_counts: Dict[str, int] = {}
            for entity in msp:
                dxftype = entity.dxftype()
                entity_counts[dxftype] = entity_counts.get(dxftype, 0) + 1

            # Bounding box calculation
            extents_box = bbox.extents(msp)
            bounding_box = None
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
                "format": "DXF",
                "dxf_version": doc.dxfversion,
                "units": doc.header.get("$INSUNITS", "Unknown"),
                "total_entities": len(msp),
                "entity_breakdown": entity_counts,
                "layer_count": len(layers),
                "layers": layers[:20],  # Limit layer output to prevent context flooding
                "bounding_box": bounding_box,
            }
        except Exception as e:
            return {"error": f"Failed to parse DXF: {str(e)}"}

    elif ext in [".step", ".stp"]:
        # Lightweight structural parsing for STEP files (pure-python fallback)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(500)]
            header_lines = [
                l.strip()
                for l in lines
                if "FILE_DESCRIPTION" in l or "FILE_NAME" in l or "SCHEMA" in l
            ]
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            return {
                "format": "STEP",
                "file_size_mb": round(file_size_mb, 2),
                "header_summary": header_lines,
                "note": "For full boundary-representation (B-Rep) solids, install cadquery or pythonocc-core.",
            }
        except Exception as e:
            return {"error": f"Failed to read STEP file: {str(e)}"}
    else:
        return {"error": f"Unsupported CAD extension: {ext}. Only DXF and STEP are supported."}


# ============================================================================
# 2. Dynamic MCP Tool Adapter & Client Bridge
# ============================================================================

def json_schema_to_pydantic(schema: Dict[str, Any], model_name: str) -> type[BaseModel]:
    """Dynamically converts a JSON Schema properties definition into a Pydantic

    model.
    """
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: Dict[str, Any] = {}
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        py_type = type_mapping.get(json_type, Any)
        description = field_info.get("description", "")
        default_val = ... if field_name in required_fields else field_info.get("default", None)

        fields[field_name] = (
            py_type if field_name in required_fields else Optional[py_type],
            Field(default=default_val, description=description),
        )

    if not fields:
        return create_model(model_name)

    return create_model(model_name, **fields)


async def mcp_tool_to_langchain_tool(
    session: ClientSession, mcp_tool: mcp_types.Tool
) -> StructuredTool:
    """Wraps an MCP tool dynamically into a LangChain StructuredTool ready for

    LLM invocation.
    """
    args_model = json_schema_to_pydantic(
        schema=mcp_tool.inputSchema,
        model_name=f"{mcp_tool.name}Input",
    )

    async def _async_runner(**kwargs) -> str:
        try:
            result: mcp_types.CallToolResult = await session.call_tool(
                name=mcp_tool.name,
                arguments=kwargs,
            )
            # Unpack tool results into a structured string
            output_parts = []
            for item in result.content:
                if isinstance(item, mcp_types.TextContent):
                    output_parts.append(item.text)
                else:
                    output_parts.append(str(item))
            return "\n".join(output_parts) if output_parts else "Tool executed successfully with no output."
        except Exception as e:
            return f"Error executing MCP tool '{mcp_tool.name}': {str(e)}"

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
        args_schema=args_model,
        coroutine=_async_runner,
    )


# ============================================================================
# 3. Connection Manager for Existing ntop MCP Server
# ============================================================================

class NTOPMCPConnectionConfig:
    """Connection settings for the existing ntop MCP server.

    Supports SSE (running daemon) or STDIO (process spawning).
    """
    def __init__(
        self,
        transport_type: Literal["sse", "stdio"] = "sse",
        # For SSE transport (ntop MCP server already listening on an HTTP/SSE port):
        sse_url: str = "http://localhost:8000/sse",
        # For stdio transport (e.g. running ntop-mcp executable/python wrapper directly):
        stdio_command: Optional[str] = "python",
        stdio_args: Optional[List[str]] = None,
        stdio_env: Optional[Dict[str, str]] = None,
    ):
        self.transport_type = transport_type
        self.sse_url = sse_url
        self.stdio_command = stdio_command
        self.stdio_args = stdio_args or ["-m", "ntop_mcp_server"]
        self.stdio_env = stdio_env


# ============================================================================
# 4. Agent Execution Loop
# ============================================================================

async def run_agent(ntop_config: NTOPMCPConnectionConfig, user_query: str):
    """Initializes the LLM, connects to ntop MCP, registers local CAD tools, and

    executes the query.
    """
    print(f"\n[+] Connecting to local Ollama LLM endpoint (http://localhost:11434)...")
    # Recommended models with strong tool-calling capabilities: 'qwen2.5:14b', 'llama3.1:8b', 'mistral-nemo'
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    print(f"[+] Establishing MCP client transport to ntop ({ntop_config.transport_type.upper()})...")

    # Select the client transport
    if ntop_config.transport_type == "sse":
        client_context = sse_client(ntop_config.sse_url)
    else:
        server_params = StdioServerParameters(
            command=ntop_config.stdio_command,
            args=ntop_config.stdio_args,
            env=ntop_config.stdio_env,
        )
        client_context = stdio_client(server_params)

    async with client_context as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize MCP Protocol Handshake
            await session.initialize()
            print("[✓] MCP connection initialized.")

            # 2. Discover tools dynamically from the ntop MCP server
            discovered_tools_response = await session.list_tools()
            ntop_tools = []
            for tool_def in discovered_tools_response.tools:
                adapted_tool = await mcp_tool_to_langchain_tool(session, tool_def)
                ntop_tools.append(adapted_tool)
                print(f"    - Discovered ntop tool: {tool_def.name} -> {tool_def.description}")

            # 3. Combine with local CAD analysis tool
            all_tools = ntop_tools + [inspect_cad_file]
            print(f"[✓] Total active tools registered: {len(all_tools)}")

            # 4. Create the LangGraph ReAct agent
            agent = create_react_agent(
                model=llm,
                tools=all_tools,
            )

            # 5. Execute query
            print(f"\n[+] Executing User Query: '{user_query}'")
            print("-" * 60)

            # LangGraph async invocation stream
            inputs = {"messages": [HumanMessage(content=user_query)]}
            async for chunk in agent.astream(inputs, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                print(f"[Agent Call] Tool: {tc['name']} | Args: {tc['args']}")
                        elif isinstance(msg, ToolMessage):
                            content_preview = str(msg.content)[:180] + "..." if len(str(msg.content)) > 180 else str(msg.content)
                            print(f"[Tool Response] -> {content_preview}")
                        elif isinstance(msg, AIMessage) and msg.content:
                            print(f"\n[Agent Final Answer]:\n{msg.content}\n")


# ============================================================================
# 5. CLI Entrypoint & Demonstration
# ============================================================================

if __name__ == "__main__":
    # Choose between SSE (if ntop MCP is running as a service) or STDIO
    # Example: If your ntop MCP server is running on http://127.0.0.1:8000/sse
    config = NTOPMCPConnectionConfig(
        transport_type="sse",
        sse_url="http://127.0.0.1:8000/sse",
    )

    # Alternate STDIO configuration:
    # config = NTOPMCPConnectionConfig(
    #     transport_type="stdio",
    #     stdio_command="python",
    #     stdio_args=["-m", "ntop_mcp_server", "--host", "localhost", "--port", "3000"],
    # )

    sample_query = (
        "1. Check ntop for the top active network flow talkers and active alerts. "
        "2. Inspect the local CAD file at 'sample_bracket.dxf' to determine its bounding dimensions and layers. "
        "3. Provide a unified summary."
    )

    try:
        asyncio.run(run_agent(config, sample_query))
    except KeyboardInterrupt:
        print("\n[!] Execution cancelled by user.")
    except Exception as err:
        print(f"\n[!] Failed to execute agent: {err}", file=sys.stderr)