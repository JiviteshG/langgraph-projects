"""Sherlock Holmes ReAct agent.

Built with LangGraph/LangChain's `create_agent`, demonstrating three concepts beyond
a plain tool-calling loop: checkpointer persistence (per-thread memory), a
`before_agent` middleware hook, and a `HumanInTheLoopMiddleware`-gated tool.
"""
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, before_agent
from langchain.messages import RemoveMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv()

SYSTEM_PROMPT = (
    "You are Sherlock Holmes. Always answer sarcastically, in character. "
    "Use the multiply tool for any calculation you're asked to perform. "
    "Once you are certain who the culprit is, call accuse_suspect to file the "
    "accusation — never just announce the suspect's name in plain text."
)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The product of the two numbers.
    """
    return a * b


def accuse_suspect(name: str) -> str:
    """Formally accuse a suspect of the crime. Final and cannot be undone — only
    call this once you are certain.

    Args:
        name: The full name of the suspect being accused.

    Returns:
        Confirmation that the accusation was filed.
    """
    return f"{name} has been formally accused and taken into custody."


@before_agent
def trim_tool_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Drop prior ToolMessages before each new turn, so tool output never piles up."""
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if not tool_messages:
        return None
    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}


def build_agent(model=None):
    """Build the compiled Sherlock Holmes agent.

    `model` is injectable so tests can supply a fake chat model instead of calling
    the live Groq API. Defaults to the real model used at runtime.
    """
    if model is None:
        model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.9)

    return create_agent(
        model=model,
        tools=[multiply, accuse_suspect],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            trim_tool_messages,
            HumanInTheLoopMiddleware(
                interrupt_on={"multiply": False, "accuse_suspect": True},
                description_prefix="Sherlock wants to make an accusation — approve before it's final",
            ),
        ],
    )
