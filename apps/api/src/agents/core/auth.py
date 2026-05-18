from contextvars import ContextVar
from functools import wraps

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.tool_node import ToolRuntime

from agents.utils.workspace import load_workspace

_workspace_id_ctx: ContextVar[str | None] = ContextVar("workspace_id", default=None)


def set_workspace_context(workspace_id: str | None) -> None:
    _workspace_id_ctx.set(workspace_id)


def get_workspace_context() -> str | None:
    return _workspace_id_ctx.get()


def authorize_tool(tool_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(
            *args,
            runtime: ToolRuntime = None,
            config: RunnableConfig = None,
            **kwargs,
        ):
            workspace_id = None
            if config and isinstance(config, dict):
                workspace_id = config.get("configurable", {}).get("workspace_id")
            if not workspace_id and runtime and hasattr(runtime, "config"):
                rc = runtime.config
                if isinstance(rc, dict):
                    workspace_id = rc.get("configurable", {}).get("workspace_id")
            if not workspace_id:
                workspace_id = get_workspace_context()

            workspace = load_workspace(workspace_id) if workspace_id else None

            if workspace and tool_name not in workspace.allowed_tools:
                return (
                    "TOOL_NOT_AVAILABLE: This tool is not enabled for your workspace."
                )

            mandatory_filters = getattr(workspace, "filters", {}) if workspace else {}
            kwargs["mandatory_filters"] = mandatory_filters
            kwargs.pop("config", None)

            return func(*args, runtime=runtime, **kwargs)

        return wrapper

    return decorator
