"""Worker subgraph: one agent attempting one task, `agent ⇄ tools`, built by hand.

The worker knows only the task prompt, the CLI's root `--help`, `TARGET_DOCS_PATH`
if set, and (docs arm) the live docs MCP. Every model turn and every tool result
is written in full to a transcript; the model sees tool output capped at
MAX_TOOL_OUTPUT characters.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tester.budget import CostMeter
from tester.docs_mcp import ALLOWED_TOOLS, DocsMCP
from tester.run_cli import CliResult, CliRunner

PROMPTS = Path(__file__).parent / "prompts"
ARMS = ("help", "docs")
MAX_TOOL_OUTPUT = 40_000
MAX_DOCS_CALLS = 10

RUN_CLI_TOOL = {
    "type": "function",
    "function": {
        "name": "run_cli",
        "description": (
            "Run the `langsmith` CLI. `args` is the argument list after `langsmith`, one argument per item, "
            'e.g. ["datasets", "list"]. Returns the exit code, stdout, and stderr.'
        ),
        "parameters": {
            "type": "object",
            "properties": {"args": {"type": "array", "items": {"type": "string"},
                                    "description": "Arguments after `langsmith`"}},
            "required": ["args"],
        },
    },
}


def mcp_tool(t: dict) -> dict:
    return {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                             "parameters": t.get("inputSchema") or {"type": "object", "properties": {}}}}


def system_prompt(root_help: str, arm: str, docs_file_text: str | None = None) -> str:
    docs_line = (" You can also search and read the LangChain documentation (which covers LangSmith) "
                 "with the docs tools." if arm == "docs" else "")
    extra = f"\nProject notes provided by the user:\n\n{docs_file_text}\n" if docs_file_text else ""
    return (PROMPTS / "tester.md").read_text().format(docs_line=docs_line, root_help=root_help.strip(),
                                                      extra_docs=extra)


def format_cli_result(res: CliResult) -> str:
    if not res.allowed:
        return f"DENIED by policy: {res.decision.reason}"
    if res.timed_out:
        return f"TIMED OUT after {res.duration_s:.0f}s (process killed)"
    parts = [f"exit code: {res.exit_code}"]
    if res.stdout:
        parts.append("stdout:\n" + res.stdout)
    if res.stderr:
        parts.append("stderr:\n" + res.stderr)
    if not res.stdout and not res.stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def cap(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n... [output truncated: {len(text):,} characters in total]"


def asks_for_json(argv: list[str]) -> bool:
    for i, a in enumerate(argv):
        if a in ("--format=json", "--format=jsonl"):
            return True
        if a == "--format" and argv[i + 1:i + 2] in (["json"], ["jsonl"]):
            return True
    return False


def text_of(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text")


@dataclass
class WorkerEnv:
    runner: CliRunner
    model: Any
    meter: CostMeter
    transcript: Path
    docs: DocsMCP | None = None
    max_turns: int = 30
    cli_results: list[CliResult] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})

    def log(self, event: str, **data) -> None:
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, **data}, default=str) + "\n")


class WorkerState(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    docs_calls: int
    stop_reason: str


def build_worker(env: WorkerEnv):
    tools = [RUN_CLI_TOOL] + ([mcp_tool(t) for t in env.docs.tools()] if env.docs else [])
    bound = env.model.bind_tools(tools)

    def agent(state: WorkerState) -> dict:
        if env.meter.exhausted:
            return {"stop_reason": "cost_ceiling"}
        if state["turns"] >= env.max_turns:
            return {"stop_reason": "max_turns"}
        msg = bound.invoke(state["messages"])
        usage = getattr(msg, "usage_metadata", None) or {}
        cost = env.meter.add(usage)
        env.usage["input_tokens"] += usage.get("input_tokens") or 0
        env.usage["output_tokens"] += usage.get("output_tokens") or 0
        env.usage["cost_usd"] += cost
        env.log("model", content=msg.content, tool_calls=msg.tool_calls, usage=usage)
        update = {"messages": [msg], "turns": state["turns"] + 1}
        if not msg.tool_calls:
            update["stop_reason"] = "answered"
        return update

    def run_tools(state: WorkerState) -> dict:
        docs_calls = state["docs_calls"]
        out = []
        for call in state["messages"][-1].tool_calls:
            name, args = call["name"], call.get("args") or {}
            if name == "run_cli":
                argv = args.get("args")
                if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
                    text = "ERROR: `args` must be a list of strings, one per argument"
                else:
                    res = env.runner.run(argv)
                    env.cli_results.append(res)
                    text = format_cli_result(res)
            elif env.docs and name in ALLOWED_TOOLS:
                if docs_calls >= MAX_DOCS_CALLS:
                    text = f"DENIED by policy: budget of {MAX_DOCS_CALLS} docs calls reached"
                else:
                    docs_calls += 1
                    try:
                        text = env.docs.call(name, args)
                    except Exception as e:  # network trouble is reported to the agent, not fatal
                        text = f"ERROR: docs tool failed: {e}"
            else:
                text = f"ERROR: unknown tool {name!r}"
            env.log("tool", name=name, args=args, output=text)
            out.append(ToolMessage(content=cap(text), tool_call_id=call["id"], name=name))
        return {"messages": out, "docs_calls": docs_calls}

    def route(state: WorkerState) -> str:
        return END if state.get("stop_reason") else "tools"

    g = StateGraph(WorkerState)
    g.add_node("agent", agent)
    g.add_node("tools", run_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile()


def run_worker(env: WorkerEnv, system: str, prompt: str) -> dict:
    env.log("start", system=system, prompt=prompt)
    graph = build_worker(env)
    final = graph.invoke(
        {"messages": [SystemMessage(system), HumanMessage(prompt)], "turns": 0, "docs_calls": 0, "stop_reason": ""},
        {"recursion_limit": env.max_turns * 2 + 10},
    )
    last_ai = next((m for m in reversed(final["messages"]) if isinstance(m, AIMessage)), None)
    answer = text_of(last_ai) if last_ai else ""
    results = env.cli_results
    summary = {
        "final_answer": answer,
        "stop_reason": final.get("stop_reason") or "unknown",
        "turns": final["turns"],
        "cli_calls": len(results),
        "help_calls": sum(1 for r in results if r.decision.is_help),
        "denied": [{"argv": r.argv, "reason": r.decision.reason} for r in results if not r.allowed],
        "errors": sum(1 for r in results if r.allowed and r.exit_code not in (0, None)),
        "used_json_format": any(asks_for_json(r.argv) for r in results if r.allowed),
        "docs_calls": final["docs_calls"],
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in env.usage.items()},
    }
    env.log("end", **summary)
    return summary
