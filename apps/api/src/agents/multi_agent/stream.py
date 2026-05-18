import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents.core.auth import set_workspace_context
from agents.core.sse import process_event
from agents.multi_agent.supervisor import build_supervisor
from core.config import config

DEFAULT_WORKSPACE_ID = "ws_test_001"


async def stream_agent(
    query: str,
    thread_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
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
        "recursion_limit": 80,
    }

    last_status: str | None = None
    seen_answers: set[str] = set()

    async with AsyncPostgresSaver.from_conn_string(config.PG_URL) as checkpointer:
        await checkpointer.setup()
        graph = build_supervisor(checkpointer=checkpointer)

        async for event in graph.astream_events(
            initial_state, config=graph_config, version="v2"
        ):
            result = process_event(event)
            if result is None:
                continue

            # Deduplicate consecutive identical status events
            if result["event"] == "status":
                if result["data"] == last_status:
                    continue
                last_status = result["data"]

            # Deduplicate answer events (supervisor may echo sub-agent answer)
            elif result["event"] == "answer":
                key = result["data"][:120]
                if key in seen_answers:
                    continue
                seen_answers.add(key)

            yield result
