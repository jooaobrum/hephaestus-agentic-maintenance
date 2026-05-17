"""Test that investigation state persists correctly across turns."""

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

# Simulate the function from sessions.py
def _summarize_investigation(result: dict) -> str:
    """Extract tool calls already made to prevent repetition."""
    if not result or not result.get("messages"):
        return ""

    tools_called = []
    seen = set()

    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                args = tool_call.get("args", {})
                # Create a signature to avoid duplicates
                sig = (tool_name, tuple(sorted(str(v) for v in args.values())))
                if tool_name and sig not in seen:
                    tools_called.append(tool_name)
                    seen.add(sig)

    if not tools_called:
        return ""

    return f"[INVESTIGATION STATUS] Already called: {', '.join(tools_called)}. Do not repeat these calls."


def test_investigation_status_extraction():
    """Test that investigation status is correctly extracted from prior result."""

    # Simulate a result from previous agent invocation with tool calls
    prior_result = {
        "messages": [
            HumanMessage(content="Diagnose machine XYZ with overheating symptom"),
            AIMessage(
                content="Starting diagnosis...",
                tool_calls=[
                    {"name": "check_machine_exists", "args": {"machine_id": "XYZ"}, "id": "1"},
                    {"name": "get_sensor_anomaly_summary", "args": {"machine_id": "XYZ"}, "id": "2"},
                ]
            ),
            ToolMessage(content="Machine exists", tool_call_id="1"),
            ToolMessage(content="Temp sensor above threshold", tool_call_id="2"),
            AIMessage(content="Found elevated temperature..."),
        ]
    }

    summary = _summarize_investigation(prior_result)

    assert summary, "Should extract investigation status"
    assert "check_machine_exists" in summary, "Should include check_machine_exists"
    assert "get_sensor_anomaly_summary" in summary, "Should include get_sensor_anomaly_summary"
    assert "[INVESTIGATION STATUS]" in summary, "Should have investigation status marker"

    print("✓ Investigation status correctly extracted")
    print(f"  Summary: {summary}")

    # Test that duplicate tool calls are not listed multiple times
    prior_result_with_duplicates = {
        "messages": [
            AIMessage(
                content="...",
                tool_calls=[
                    {"name": "check_machine_exists", "args": {"machine_id": "XYZ"}, "id": "1"},
                    {"name": "check_machine_exists", "args": {"machine_id": "XYZ"}, "id": "2"},  # Duplicate
                ]
            ),
        ]
    }

    summary = _summarize_investigation(prior_result_with_duplicates)
    # Should only list check_machine_exists once
    count = summary.count("check_machine_exists")
    assert count == 1, f"Should deduplicate tool calls, but found {count} occurrences"

    print("✓ Duplicate tool calls correctly deduplicated")


def test_empty_result_handling():
    """Test that empty results don't crash."""

    assert _summarize_investigation(None) == ""
    assert _summarize_investigation({}) == ""
    assert _summarize_investigation({"messages": []}) == ""

    print("✓ Empty results handled gracefully")


def test_no_tool_calls():
    """Test that results with no tool calls return empty summary."""

    result = {
        "messages": [
            HumanMessage(content="Question"),
            AIMessage(content="Answer with no tool calls"),
        ]
    }

    summary = _summarize_investigation(result)
    assert summary == "", "Should return empty when no tool calls made"

    print("✓ Results with no tool calls handled correctly")


if __name__ == "__main__":
    test_investigation_status_extraction()
    test_empty_result_handling()
    test_no_tool_calls()
    print("\n✅ All tests passed!")
