"""Translates LangGraph astream_events (v2) into SSE event dicts.

SSE event shapes consumed by the Streamlit UI:
  {"event": "trace_id",   "trace_id": str}
  {"event": "status",     "data": str}
  {"event": "reasoning",  "data": str}
  {"event": "tool_calls", "data": [str]}
  {"event": "answer",     "data": str}

Reasoning extraction (option 2): on every inner LLM turn within an owned
subgraph (troubleshooting / summarizer / supervisor), inspect the completed
AIMessage at `on_chain_end` of the `agent` node. If the message has pending
tool calls, its text content is the model's pre-tool rationale → emit as
`reasoning`. If it has no tool calls, it's the final reply → emit as `answer`
(deduped upstream by stream.py).
"""

from langchain_core.messages import AIMessage

from core.config import app

_NODE_STATUS: dict[str, str] = app.sse.node_status
_TOKEN_NODES: frozenset[str] = frozenset(app.sse.token_nodes)
_ANSWER_NODES: frozenset[str] = frozenset(app.sse.answer_nodes)
_TRANSFER_SCAN_LEN = app.agents.transfer_message_scan_length

_KNOWN_OWNERS = _TOKEN_NODES | _ANSWER_NODES | frozenset(_NODE_STATUS)


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in content
        )
    return ""


def _match_owner(node: str, ns: str) -> str:
    """Pick the owning agent/supervisor for an event.

    Inner react-agent LLM nodes have node="agent" or "tools"; the owning subgraph
    name lives in `langgraph_checkpoint_ns`, e.g. "troubleshooting:abc|agent:xyz".
    """
    if node in _KNOWN_OWNERS:
        return node
    for segment in ns.split("|"):
        name = segment.split(":", 1)[0]
        if name in _KNOWN_OWNERS:
            return name
    return node


_SUBAGENT_NODES = frozenset(app.sse.answer_nodes) - {"supervisor"}


def make_event_processor():
    """Return a stateful `process_event` closure for one stream."""
    emitted_reasoning: set[str] = set()
    # Set to True after a subagent finishes; causes the next supervisor answer
    # turn (the relay) to be suppressed — the subagent's answer already reached
    # the UI via the subagent on_chain_end handler above.
    state = {"suppress_next_supervisor_answer": False}

    def process_event(event) -> list[dict]:
        evt_type = event.get("event", "")
        metadata = event.get("metadata", {})
        node = metadata.get("langgraph_node", "")
        ns = metadata.get("langgraph_checkpoint_ns", "") or ""
        owner = _match_owner(node, ns)
        results: list[dict] = []

        if (
            evt_type == "on_chain_start"
            and node in _NODE_STATUS
            and "langgraph_triggers" in metadata
        ):
            results.append({"event": "status", "data": _NODE_STATUS[node]})
            return results

        if evt_type == "on_tool_start" and owner in _TOKEN_NODES:
            tool_name = event.get("name", "")
            if tool_name:
                results.append({"event": "tool_calls", "data": [tool_name]})
            return results

        # Inner react-agent LLM turn finished: pull reasoning / answer off the AIMessage.
        if evt_type == "on_chain_end" and node == "agent" and owner in _TOKEN_NODES:
            msg = _last_ai_message(event["data"].get("output"))
            if msg is not None:
                text = _text_from_content(msg.content).strip()
                msg_key = msg.id or text[:80]
                if text and msg_key not in emitted_reasoning:
                    emitted_reasoning.add(msg_key)
                    has_tool_calls = bool(getattr(msg, "tool_calls", None))
                    if has_tool_calls:
                        results.append({"event": "reasoning", "data": text})
            return results

        # Mark that a subagent just finished so we can suppress the supervisor relay.
        if evt_type == "on_chain_end" and node in _SUBAGENT_NODES:
            state["suppress_next_supervisor_answer"] = True

        # Subgraph finished: emit the final answer.
        # Supervisor relay turns (the LLM turn that runs after a subagent returns)
        # are suppressed — the subagent's own answer already reached the UI.
        if evt_type == "on_chain_end" and node in _ANSWER_NODES:
            output = event["data"].get("output") or {}
            if isinstance(output, dict):
                for msg in reversed(output.get("messages", [])):
                    if isinstance(msg, AIMessage):
                        text = _text_from_content(msg.content).strip()
                        if not text:
                            break
                        if "transfer" in text.lower()[:_TRANSFER_SCAN_LEN]:
                            break
                        if node == "supervisor":
                            if state["suppress_next_supervisor_answer"]:
                                state["suppress_next_supervisor_answer"] = False
                                break
                        results.append({"event": "answer", "data": text})
                        break
            return results

        return results

    return process_event


def _last_ai_message(output) -> AIMessage | None:
    """Extract the last AIMessage from a node's output payload."""
    if output is None:
        return None
    if isinstance(output, AIMessage):
        return output
    if isinstance(output, dict):
        msgs = output.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                return msg
    return None


# Backwards-compatible single-event API.
def process_event(event: dict) -> dict | None:
    proc = make_event_processor()
    events = proc(event)
    return events[0] if events else None
