import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents.core.auth import set_workspace_context
from agents.core.llm import make_llm
from agents.core.sse import make_event_processor
from agents.multi_agent.loader import AGENT_NAMES, build_agent
from agents.multi_agent.supervisor import build_supervisor
from core.config import app, secrets


async def stream_agent(
    query: str,
    thread_id: str,
    workspace_id: str = app.agents.default_workspace_id,
    mode: str = "auto",
):
    """Async generator yielding SSE event dicts for the RCA supervisor graph."""
    run_id = uuid.uuid4()
    yield {"event": "trace_id", "trace_id": str(run_id)}

    set_workspace_context(workspace_id)

    initial_state = {"messages": [HumanMessage(content=query)]}
    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "workspace_id": workspace_id,
        },
        "run_id": run_id,
        "recursion_limit": app.agents.recursion_limit,
    }

    last_status: str | None = None
    seen_answers: set[str] = set()

    async with AsyncPostgresSaver.from_conn_string(secrets.PG_URL) as checkpointer:
        await checkpointer.setup()
        if mode in AGENT_NAMES:
            graph = build_agent(mode, make_llm(), checkpointer=checkpointer)
        else:
            graph = build_supervisor(checkpointer=checkpointer)

        process_event = make_event_processor()
        async for event in graph.astream_events(
            initial_state, config=graph_config, version="v2"
        ):
            evt_type = event.get("event", "")
            node = event.get("metadata", {}).get("langgraph_node", "")
            if evt_type == "on_chain_end" and node in (
                "troubleshooting",
                "summarizer",
                "supervisor",
            ):
                import logging

                logging.getLogger(__name__).warning("SSE_DEBUG chain_end node=%s", node)
            for result in process_event(event):
                if result["event"] == "status":
                    if result["data"] == last_status:
                        continue
                    last_status = result["data"]

                # Supervisor may echo the sub-agent's final answer — dedupe by prefix.
                elif result["event"] == "answer":
                    key = result["data"][: app.agents.answer_dedup_key_length]
                    if key in seen_answers:
                        continue
                    seen_answers.add(key)

                yield result
