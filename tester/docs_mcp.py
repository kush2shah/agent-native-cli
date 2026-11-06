"""Minimal client for the live LangChain docs MCP server (the `docs` arm).

Only the two read tools are exposed to the worker. `submit_feedback` sends
messages to the docs team, so it is filtered out of the tool list and refused
if called anyway.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable

DOCS_MCP_URL = "https://docs.langchain.com/mcp"
ALLOWED_TOOLS = ("search_docs_by_lang_chain", "query_docs_filesystem_docs_by_lang_chain")

Post = Callable[[str, dict, dict], tuple[dict, str]]  # (url, headers, body) -> (response headers, text)


def http_post(url: str, headers: dict, body: dict) -> tuple[dict, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return dict(resp.headers), resp.read().decode("utf-8", "replace")


def _parse(text: str) -> dict:
    """The server answers with JSON or a one-event SSE stream."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise ValueError(f"unexpected MCP response: {text[:200]!r}")


class DocsMCP:
    def __init__(self, url: str = DOCS_MCP_URL, post: Post = http_post):
        self.url = url
        self.post = post
        self.session: str | None = None
        self._id = 0
        self._tools: list[dict] | None = None

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        resp_headers, text = self.post(self.url, headers, {"jsonrpc": "2.0", "id": self._id, "method": method,
                                                            "params": params or {}})
        sid = {k.lower(): v for k, v in resp_headers.items()}.get("mcp-session-id")
        if sid:
            self.session = sid
        msg = _parse(text)
        if "error" in msg:
            raise RuntimeError(f"MCP {method} failed: {msg['error']}")
        return msg.get("result", {})

    def tools(self) -> list[dict]:
        """The allowed tools, with the server's own names, descriptions, and schemas."""
        if self._tools is None:
            if self.session is None:
                self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "cli-tester", "version": "0.1"}})
            listed = self._rpc("tools/list").get("tools", [])
            self._tools = [t for t in listed if t["name"] in ALLOWED_TOOLS]
        return self._tools

    def call(self, name: str, arguments: dict) -> str:
        if name not in ALLOWED_TOOLS:
            raise PermissionError(f"docs tool {name!r} is not allowed")
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        text = "\n\n".join(parts)
        return f"ERROR: {text}" if result.get("isError") else text
