"""Tests for the Sherlock Holmes LangGraph agent.

The LLM is stubbed with FakeMessagesListChatModel so these run with no network
access and no API key.
"""
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from agent import accuse_suspect, build_agent, multiply, trim_tool_messages


class FakeToolCallingModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel, but usable inside create_agent.

    The base class's inherited `bind_tools` raises NotImplementedError (it's meant
    to be overridden by real provider models). Since this fake ignores whatever is
    bound anyway and just replays its scripted `responses`, binding is a no-op.
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_multiply():
    assert multiply(3, 4) == 12


def test_accuse_suspect():
    result = accuse_suspect("Moriarty")
    assert "Moriarty" in result
    assert "accused" in result.lower()


def test_trim_tool_messages_removes_tool_messages():
    state = {
        "messages": [
            HumanMessage(content="hi", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"name": "multiply", "args": {"a": 1, "b": 2}, "id": "c1"}],
            ),
            ToolMessage(content="2", tool_call_id="c1", id="tm1"),
        ]
    }
    # @before_agent turns the function into an AgentMiddleware instance whose
    # logic lives on the .before_agent method — not directly callable itself.
    result = trim_tool_messages.before_agent(state, None)
    assert result is not None
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], RemoveMessage)
    assert result["messages"][0].id == "tm1"


def test_trim_tool_messages_returns_none_when_nothing_to_trim():
    state = {"messages": [HumanMessage(content="hi", id="h1")]}
    result = trim_tool_messages.before_agent(state, None)
    assert result is None


def test_build_agent_compiles():
    fake_model = FakeToolCallingModel(responses=[AIMessage(content="hello")])
    agent = build_agent(model=fake_model)
    assert agent is not None


def test_checkpointer_persists_across_invocations_on_same_thread():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(content="First answer."),
            AIMessage(content="Second answer."),
        ]
    )
    agent = build_agent(model=fake_model)
    config = {"configurable": {"thread_id": "t1"}}

    agent.invoke({"messages": [HumanMessage(content="Hello, I'm Watson.")]}, config=config)
    result = agent.invoke({"messages": [HumanMessage(content="What's my name?")]}, config=config)

    # both turns' human + ai messages show up in state -> persisted, not reset
    assert len(result["messages"]) == 4


def test_checkpointer_isolates_different_threads():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(content="First answer."),
            AIMessage(content="Second answer."),
        ]
    )
    agent = build_agent(model=fake_model)

    agent.invoke(
        {"messages": [HumanMessage(content="Hello, I'm Watson.")]},
        config={"configurable": {"thread_id": "t1"}},
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config={"configurable": {"thread_id": "t2"}},
    )

    # thread t2 never saw thread t1's turn
    assert len(result["messages"]) == 2


def test_hitl_interrupts_before_accuse_suspect():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "accuse_suspect", "args": {"name": "Moriarty"}, "id": "call_1"}
                ],
            ),
        ]
    )
    agent = build_agent(model=fake_model)
    config = {"configurable": {"thread_id": "hitl-test"}}

    response = agent.invoke({"messages": [HumanMessage(content="Accuse Moriarty.")]}, config=config)

    assert "__interrupt__" in response


def test_multiply_executes_without_interrupt():
    fake_model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "multiply", "args": {"a": 3, "b": 4}, "id": "call_1"}],
            ),
            AIMessage(content="Elementary -- it's 12."),
        ]
    )
    agent = build_agent(model=fake_model)
    config = {"configurable": {"thread_id": "multiply-test"}}

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is 3 times 4?")]}, config=config
    )

    assert "__interrupt__" not in response
    assert response["messages"][-1].content == "Elementary -- it's 12."
