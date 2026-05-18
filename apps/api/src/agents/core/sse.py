"""Translates LangGraph astream_events (v2) into SSE event dicts.

SSE event shapes consumed by the Streamlit UI:
  {"event": "trace_id",   "trace_id": str}
  {"event": "status",     "data": str}
  {"event": "token",      "data": str}
  {"event": "tool_calls", "data": [str]}
  {"event": "answer",     "data": str}
"""

_NODE_STATUS: dict[str, str] = {
    "supervisor": "Coordinator is routing...",
    "troubleshooting": "Troubleshooting agent is working...",
    "summarizer": "Summarizer agent is working...",
}

# Only stream tokens and tool calls from sub-agent nodes
_TOKEN_NODES = {"troubleshooting", "summarizer"}

# Only extract final answer from sub-agent nodes (supervisor echoes = duplication)
_ANSWER_NODES = {"troubleshooting", "summarizer"}


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in content
        )
    return ""


def process_event(event: dict) -> dict | None:
    evt_type = event.get("event", "")
    metadata = event.get("metadata", {})
    node = metadata.get("langgraph_node", "")

    # Status: only fire on top-level graph node entry, identified by langgraph_triggers.
    # Inner agent chains re-enter nodes without this metadata key, causing duplicates.
    if (
        evt_type == "on_chain_start"
        and node in _NODE_STATUS
        and "langgraph_triggers" in metadata
    ):
        return {"event": "status", "data": _NODE_STATUS[node]}

    # Streaming tokens — sub-agent nodes only
    if evt_type == "on_chat_model_stream" and node in _TOKEN_NODES:
        chunk = event["data"].get("chunk")
        if chunk:
            text = _text_from_content(chunk.content)
            if text:
                return {"event": "token", "data": text}

    # Tool invocation — sub-agent nodes only
    if evt_type == "on_tool_start" and node in _TOKEN_NODES:
        tool_name = event.get("name", "")
        if tool_name:
            return {"event": "tool_calls", "data": [tool_name]}

    # Final answer — sub-agent nodes only
    if evt_type == "on_chain_end" and node in _ANSWER_NODES:
        from langchain_core.messages import AIMessage

        output = event["data"].get("output") or {}
        if isinstance(output, dict):
            for msg in reversed(output.get("messages", [])):
                if isinstance(msg, AIMessage):
                    text = _text_from_content(msg.content).strip()
                    if text and "transfer" not in text.lower()[:30]:
                        return {"event": "answer", "data": text}

    return None
