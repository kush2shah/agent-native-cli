"""LangSmith API proxy for exported Harbor tasks. It holds the target key, so the agent's container never does.

In Harbor the agent gets a real shell, so `run_cli`'s allowlist can't protect the
workspace there. Instead, the agent's container points the CLI at this service
(`LANGSMITH_ENDPOINT=http://lsproxy:8080`) with a dummy key. For each request the proxy:

- drops the client's credentials and adds the real key (and workspace, if set)
- forwards reads only: GET/HEAD, plus POSTs to query-style endpoints (`.../query`,
  `.../stats`, `.../search`, `.../count`), which LangSmith uses for filtered listings
- answers everything else with a 403 whose detail names the sandbox, so it can't be
  mistaken for a LangSmith permission error
- never follows redirects, so the key can't be carried to another host
- logs one JSON line (no bodies) to `LSPROXY_LOG`

Exported tasks are read-only (see `tester/harbor/export.py`), so any write is a deny.
The POST allowlist fails closed: a read endpoint it doesn't know shows up as a
`denied` line in the log, and the fix is widening READ_POST_SUFFIXES.

Stdlib only: it runs in a bare python image and in the tests.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

READ_METHODS = ("GET", "HEAD")
READ_POST_SUFFIXES = ("/query", "/stats", "/search", "/count")
HEALTH_PATH = "/__lsproxy/health"

# Never forwarded: credentials and workspace come from the proxy, hop-by-hop headers are per connection.
_DROP_REQUEST = {"host", "x-api-key", "authorization", "x-tenant-id", "cookie", "proxy-authorization",
                 "connection", "keep-alive", "te", "trailer", "transfer-encoding", "upgrade", "content-length"}
_DROP_RESPONSE = {"connection", "keep-alive", "transfer-encoding", "trailer", "upgrade", "content-length",
                  "set-cookie"}


def decide(method: str, target: str) -> tuple[bool, str]:
    """Whether to forward a request, and why. `target` is the raw request target."""
    if not target.startswith("/"):
        return False, "only paths are proxied, not absolute URLs"
    if method in READ_METHODS:
        return True, "read"
    path = urlsplit(target).path.rstrip("/")
    if method == "POST" and path.endswith(READ_POST_SUFFIXES):
        return True, "read (query endpoint)"
    return False, f"{method} {path or '/'} would change the workspace, and this task is read-only"


def _json(status: int, detail: str) -> tuple[int, dict, bytes]:
    return status, {"Content-Type": "application/json"}, json.dumps({"detail": detail}).encode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # hand the 3xx back to the client instead of following it with our key


class _Handler(BaseHTTPRequestHandler):
    server: "ProxyServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # the JSON log replaces the default stderr lines
        pass

    def _handle(self) -> None:
        start = time.monotonic()
        method, target = self.command, self.path
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if target == HEALTH_PATH:
            self._send(200, {"Content-Type": "application/json"}, b'{"ok": true}')
            return
        allowed, reason = decide(method, target)
        if allowed:
            status, headers, data = self._forward(method, target, body)
        else:
            status, headers, data = _json(403, f"blocked by the eval sandbox, not by LangSmith: {reason}")
        # log before replying, so the line exists by the time the client sees the response
        self.server.log({"ts": time.time(), "method": method, "path": target, "allowed": allowed, "reason": reason,
                         "status": status, "bytes_in": len(body), "bytes_out": len(data),
                         "ms": round((time.monotonic() - start) * 1000)})
        self._send(status, headers, b"" if method == "HEAD" else data)

    def _forward(self, method: str, target: str, body: bytes) -> tuple[int, dict, bytes]:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST}
        headers["x-api-key"] = self.server.api_key
        if self.server.workspace_id:
            headers["X-Tenant-Id"] = self.server.workspace_id
        req = urllib.request.Request(self.server.upstream + target, data=body if body else None,
                                     headers=headers, method=method)
        try:
            with self.server.opener.open(req, timeout=self.server.timeout_s) as resp:
                status, resp_headers, data = resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as e:  # 4xx/5xx/3xx from upstream: pass through as-is
            status, resp_headers, data = e.code, e.headers, e.read()
        except Exception as e:  # upstream unreachable: say so, in the same JSON shape
            return _json(502, f"eval sandbox proxy couldn't reach LangSmith: {type(e).__name__}")
        return status, {k: v for k, v in resp_headers.items() if k.lower() not in _DROP_RESPONSE}, data

    def _send(self, status: int, headers: dict, data: bytes) -> int:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        return status

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _handle


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, upstream: str, api_key: str, workspace_id: str = "", log_path: str | None = None,
                 timeout_s: float = 60, echo: bool = True):
        super().__init__(addr, _Handler)
        self.echo = echo  # also print each log line, for `docker logs`
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.timeout_s = timeout_s
        self.opener = urllib.request.build_opener(_NoRedirect)
        self.log_path = log_path
        self._lock = threading.Lock()

    def log(self, record: dict) -> None:
        line = json.dumps(record)
        with self._lock:
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
            if self.echo:
                print(line, flush=True)


def main() -> int:
    key = os.environ.get("LSPROXY_API_KEY", "")
    if not key:
        print("LSPROXY_API_KEY is not set: export TARGET_API_KEY before `harbor run`", file=sys.stderr)
        return 1
    log_path = os.environ.get("LSPROXY_LOG", "/var/log/lsproxy/requests.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    open(log_path, "a").close()  # exists even if the agent never calls the API, so artifact collection finds it
    server = ProxyServer(("0.0.0.0", int(os.environ.get("LSPROXY_PORT", "8080"))),
                         upstream=os.environ.get("LSPROXY_UPSTREAM") or "https://api.smith.langchain.com",
                         api_key=key, workspace_id=os.environ.get("LSPROXY_WORKSPACE_ID", ""), log_path=log_path)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
