"""Test verification for DeepSeek-R1 <think> tag middleware."""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage
from src.think_middleware import extract_json_payload, sanitize_ai_message, strip_think_tags


def test_closed_think_tag():
    text = (
        "<think>\n"
        "User wants to auto-orient and slice the bracket.\n"
        "I will call the auto_orient_stl tool.\n"
        "</think>\n"
        '{"name": "auto_orient_stl", "arguments": {"stl_filepath": "test.stl"}}'
    )
    clean, thought = strip_think_tags(text)
    assert "User wants to auto-orient" in thought
    assert "<think>" not in clean
    assert "auto_orient_stl" in clean

    msg = AIMessage(content=text)
    sanitized = sanitize_ai_message(msg)
    assert len(sanitized.tool_calls) == 1
    assert sanitized.tool_calls[0]["name"] == "auto_orient_stl"
    assert sanitized.tool_calls[0]["args"]["stl_filepath"] == "test.stl"
    assert sanitized.content == ""


def test_markdown_json_block():
    text = (
        "<think>\n"
        "Let me format as markdown json block.\n"
        "</think>\n"
        "```json\n"
        '{\n  "tool": "slice_stl_to_gcode",\n  "tool_input": {"stl_filepath": "oriented.stl"}\n}\n'
        "```"
    )
    msg = AIMessage(content=text)
    sanitized = sanitize_ai_message(msg)
    assert len(sanitized.tool_calls) == 1
    assert sanitized.tool_calls[0]["name"] == "slice_stl_to_gcode"
    assert sanitized.tool_calls[0]["args"]["stl_filepath"] == "oriented.stl"


def test_regular_text_with_think():
    text = (
        "<think>\n"
        "No more tools needed, I will summarize.\n"
        "</think>\n"
        "The manufacturing pipeline completed successfully. The file is saved at output.gcode."
    )
    msg = AIMessage(content=text)
    sanitized = sanitize_ai_message(msg)
    assert len(sanitized.tool_calls) == 0
    assert "manufacturing pipeline completed successfully" in sanitized.content
    assert "<think>" not in sanitized.content


if __name__ == "__main__":
    test_closed_think_tag()
    test_markdown_json_block()
    test_regular_text_with_think()
    print("ALL MIDDLEWARE TESTS PASSED!")
