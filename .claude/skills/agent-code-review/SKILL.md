---
name: agent-code-review
description: Review LangChain/LangGraph agent code for production patterns — checkpointer persistence, correct middleware hook placement, state/context separation, human-in-the-loop gating, and tool error handling. Use when reviewing or writing create_agent-based agent code.
---

# Agent Code Review Checklist

Use this checklist when reviewing or writing a LangChain `create_agent` / LangGraph agent, to catch the gap between "it runs" and "it's production-shaped."

## 1. Persistence — is state actually being saved?
- Any agent that needs to remember context across turns must have a `checkpointer` passed to `create_agent` (`InMemorySaver` for dev/tests, `SqliteSaver`/`PostgresSaver` for real deployments).
- Every invocation needs a `thread_id` in `config` — without it, checkpointing silently does nothing.
- Flag: state stored in a global variable or passed manually between calls instead of the checkpointer + thread_id pattern.

## 2. Middleware — is the hook at the right altitude?
Four distinct hook points exist — using the wrong one is a bug, not just a style choice:
- `@before_agent` / `@after_agent` — runs ONCE per full request, before/after the whole graph. Right place for: auth gates, trimming stale messages, logging.
- `@wrap_model_call` — runs on EVERY individual LLM call within a request. Right place for: swapping model/tools dynamically mid-conversation (`request.override(...)`).
- `@dynamic_prompt` — runs on EVERY LLM call, returns a system prompt string. Right place for: prompts that need to change based on state/context mid-conversation.
- Tool-call-wrapping middleware (e.g. `HumanInTheLoopMiddleware`) — runs around EACH tool call. Right place for: per-tool approval gating.
- Flag: logic that needs to happen per-LLM-call incorrectly placed in `before_agent` (it won't re-fire), or a bare `interrupt()` scattered in a graph node instead of `interrupt_on` on a middleware.

## 3. State vs Context — don't conflate them
- **State**: read/write, persisted via checkpointer. Subclass `AgentState`, declare via `create_agent(state_schema=...)`, read/write via `runtime.state[...]` or a tool returning `Command(update={...})`.
- **Context**: read-only, per-invocation, NOT persisted. A `@dataclass`, declared via `create_agent(context_schema=...)`, passed via `agent.invoke(..., context=...)`, read via `runtime.context.field`.
- Flag: config-like values (API keys, user role, feature flags) pushed into State when they belong in Context, or mutable conversation data treated as Context.

## 4. Human-in-the-loop — gated, not blanket
- Risky/irreversible tools (sending an email, making a purchase, deleting something) should be gated with `HumanInTheLoopMiddleware(interrupt_on={"tool_name": True})`.
- Low-risk, reversible tools (a lookup, a calculation) should stay ungated (`False`) — gating everything creates approval fatigue and defeats the point.
- Resume path must use `Command(resume={"decisions": [...]})`, not a fresh `.invoke()`.

## 5. Tool error handling
- Tool functions should catch and return errors as a string result, not raise — an uncaught exception crashes the whole graph run instead of giving the agent a chance to retry or explain the failure.

## 6. Tests
- At minimum: one test proving persistence works (same `thread_id` → history carries over), one proving thread isolation (different `thread_id` → no bleed), and one proving HITL actually interrupts on the gated tool.
- Use a scripted/fake chat model (e.g. `FakeMessagesListChatModel` subclassed to no-op `bind_tools`) so tests run without a live API key or network call.

---

This checklist documents the same patterns already applied in this repo (`agent.py`, `main.py`) — `InMemorySaver`, `before_agent` trimming stale tool messages, `HumanInTheLoopMiddleware` gating only `accuse_suspect` — generalized so they can be checked against any new agent, not just this one.
