from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.config import config


async def get_checkpointer() -> AsyncPostgresSaver:
    checkpointer = AsyncPostgresSaver.from_conn_string(config.PG_URL)
    return checkpointer
