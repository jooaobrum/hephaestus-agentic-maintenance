from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from agents.core.prompts import load_prompt
from agents.multi_agent.tool_groups import TROUBLESHOOTING_TOOLS, SUMMARIZER_TOOLS
from core.config import config

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.GENERATION_MODEL,
        temperature=0,
        timeout=90,
        max_retries=2,
        streaming=True,
    )


def build_supervisor(checkpointer=None):
    llm = _make_llm()

    troubleshooting_agent = create_react_agent(
        model=llm,
        tools=TROUBLESHOOTING_TOOLS,
        prompt=load_prompt(
            _PROMPTS_DIR / "rca_troubleshooting.yml", "rca_troubleshooting"
        ),
        name="troubleshooting",
    )

    summarizer_agent = create_react_agent(
        model=llm,
        tools=SUMMARIZER_TOOLS,
        prompt=load_prompt(_PROMPTS_DIR / "summarizer.yml", "summarizer"),
        name="summarizer",
    )

    graph = create_supervisor(
        model=llm,
        agents=[troubleshooting_agent, summarizer_agent],
        prompt=load_prompt(_PROMPTS_DIR / "coordinator.yml", "coordinator"),
    ).compile(checkpointer=checkpointer)

    return graph
