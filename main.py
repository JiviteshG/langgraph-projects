"""Demo runner for the Sherlock Holmes agent.

Two demos: (1) checkpointer persistence — same thread_id recalls prior turns, a
different thread_id starts clean; (2) human-in-the-loop — accuse_suspect pauses for
approval before it executes.
"""
import sys
from pprint import pprint

from langchain.messages import HumanMessage
from langgraph.types import Command

from agent import build_agent

# Windows terminals often default to a legacy codepage (e.g. cp1252) that can't
# encode characters like em-dashes the model may output. Force UTF-8 for stdout.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def demo_persistence(agent):
    print("=" * 80)
    print("PERSISTENCE DEMO — same thread_id remembers, a new thread_id starts clean")
    print("=" * 80)

    thread_a = {"configurable": {"thread_id": "case-221b"}}

    result = agent.invoke(
        {"messages": [HumanMessage(content="My name is Watson. What is 12 times 11?")]},
        config=thread_a,
    )
    print("\n[thread case-221b, turn 1]")
    result["messages"][-1].pretty_print()

    result = agent.invoke(
        {"messages": [HumanMessage(content="What did I say my name was?")]},
        config=thread_a,
    )
    print("\n[thread case-221b, turn 2 — should recall 'Watson']")
    result["messages"][-1].pretty_print()

    thread_b = {"configurable": {"thread_id": "case-baskerville"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="What did I say my name was?")]},
        config=thread_b,
    )
    print("\n[thread case-baskerville, turn 1 — different thread, should NOT know 'Watson']")
    result["messages"][-1].pretty_print()


def demo_human_in_the_loop(agent):
    print("\n" + "=" * 80)
    print("HUMAN-IN-THE-LOOP DEMO — accuse_suspect pauses for approval")
    print("=" * 80)

    thread = {"configurable": {"thread_id": "case-hound"}}

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "The footprints, the cigar ash, the missing will — it is "
                        "obviously Jack Stapleton. Accuse him now."
                    )
                )
            ]
        },
        config=thread,
    )

    if "__interrupt__" in response:
        request = response["__interrupt__"][0].value["action_requests"][0]
        print("\n[PAUSED] Sherlock is waiting for approval before this action executes:")
        pprint(request)

        response = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=thread,
        )
        print("\n[APPROVED — accusation filed]")
        response["messages"][-1].pretty_print()
    else:
        print("\n(No interrupt raised — the model didn't call accuse_suspect this run.)")
        response["messages"][-1].pretty_print()


def main():
    agent = build_agent()
    demo_persistence(agent)
    demo_human_in_the_loop(agent)


if __name__ == "__main__":
    main()
