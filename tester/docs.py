"""Docs: snapshot the docs site, check its CLI examples against the inventory, search it.

The snapshot follows Mintlify's `llms.txt` indexes recursively and keeps every
`/langsmith/` page as markdown, so a run can say exactly which docs it used.

The docs check is static: it never runs the CLI. Each `langsmith ...` command found
in a code block or inline code span is split with shlex and resolved with the same
resolver `run_cli` uses, then compared against the command's help.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tester.inventory import Inventory
from tester.run_cli import resolve

USER_AGENT = "agent-native-cli-tester (docs snapshot)"
MAX_INDEXES = 200

# `- [Title](url): description` lines in llms.txt indexes
_INDEX_LINK = re.compile(r"^\s*-\s*\[(?P<title>[^\]]*)\]\((?P<url>[^)\s]+)\)(?::\s*(?P<desc>.*))?$")


def keep_page(url: str) -> bool:
    return "/langsmith/" in url and url.endswith(".md")


def check_page(url: str) -> bool:
    # Changelogs describe other versions; their commands aren't instructions to follow.
    return "changelog" not in url


def follow_index(url: str) -> bool:
    # 596 generated REST API reference pages: not CLI docs, and they'd swamp search.
    return "/_llms/" in url and url.endswith(".txt") and "lang-smith-rest-api" not in url


def http_get(url: str, timeout: float = 30, retries: int = 2) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1 + attempt)
    raise AssertionError("unreachable")


def _absolute(url: str, base: str) -> str:
    if url.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", base)
        return (m.group(1) if m else "") + url
    return url


def page_file(url: str) -> str:
    path = re.sub(r"^https?://[^/]+/", "", url)
    return re.sub(r"[^\w.-]+", "__", path)


def snapshot(out_dir: Path, index_url: str, fetch: Callable[[str], str] = http_get, workers: int = 12) -> dict:
    """Fetch the indexes, then every kept page. Writes pages/ and index.json under out_dir."""
    pages: dict[str, dict] = {}
    errors: list[dict] = []
    seen: set[str] = set()
    queue = [index_url]
    while queue and len(seen) < MAX_INDEXES:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            text = fetch(url)
        except Exception as e:
            errors.append({"url": url, "error": f"{type(e).__name__}: {e}"})
            continue
        for line in text.splitlines():
            m = _INDEX_LINK.match(line)
            if not m:
                continue
            link = _absolute(m["url"], url)
            if follow_index(link):
                queue.append(link)
            elif keep_page(link) and link not in pages:
                pages[link] = {"url": link, "title": m["title"].strip(), "description": (m["desc"] or "").strip()}

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    def get_page(meta: dict) -> None:
        try:
            body = fetch(meta["url"])
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"
            return
        meta["file"] = page_file(meta["url"])
        meta["chars"] = len(body)
        (pages_dir / meta["file"]).write_text(body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(get_page, pages.values()))

    for meta in pages.values():
        if "error" in meta:
            errors.append({"url": meta["url"], "error": meta.pop("error")})
    data = {
        "source": index_url,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "indexes": len(seen),
        "pages": sorted((p for p in pages.values() if "file" in p), key=lambda p: p["url"]),
        "errors": errors,
    }
    (out_dir / "index.json").write_text(json.dumps(data, indent=2))
    return data


def load_snapshot(snapshot_dir: Path) -> dict:
    return json.loads((snapshot_dir / "index.json").read_text())


# --- extracting CLI examples -------------------------------------------------

_FENCE = re.compile(r"^\s*(```+|~~~+)\s*(?P<lang>[\w+-]*)")
SHELL_LANGS = {"", "bash", "sh", "shell", "zsh", "console", "terminal", "shellscript", "text", "txt"}
_INLINE = re.compile(r"`(langsmith [^`\n]+)`")
_PROMPT = re.compile(r"^(?:\$|%|>)\s+")


def extract_commands(markdown: str) -> list[dict]:
    """`langsmith ...` commands in code blocks and inline code: {line, where, text}."""
    found: list[dict] = []
    in_fence, shell = False, False
    pending, pending_line = "", 0
    for n, raw in enumerate(markdown.splitlines(), start=1):
        if m := _FENCE.match(raw):
            in_fence = not in_fence
            shell = in_fence and m["lang"].lower() in SHELL_LANGS
            pending = ""
            continue
        if not in_fence:
            found += [{"line": n, "where": "inline", "text": m.group(1).strip()} for m in _INLINE.finditer(raw)]
            continue
        if not shell:  # python, go, hcl, json ...: `langsmith` there isn't the CLI
            continue
        line = raw.strip()
        if not pending:
            pending_line = n
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        logical = (pending + line).strip()
        if _unclosed_quote(logical) and n - pending_line < 10:
            pending = logical + " "  # a quoted argument spans lines
            continue
        pending = ""
        logical = _PROMPT.sub("", logical)
        for segment in split_shell(logical):
            segment = segment.strip()
            if segment.split(" ", 1)[0] == "langsmith":
                found.append({"line": pending_line, "where": "code", "text": segment})
    return found


def split_shell(text: str) -> list[str]:
    """Split on `&&`, `||`, `;`, and `|` outside quotes."""
    parts, buf, quote, i = [], [], "", 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i + 1 < len(text):
                buf.append(ch)
                i += 1
                ch = text[i]
        elif ch in "'\"":
            quote = ch
        elif text.startswith(("&&", "||"), i):
            parts.append("".join(buf))
            buf, i = [], i + 2
            continue
        elif ch in ";|":
            parts.append("".join(buf))
            buf, i = [], i + 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _unclosed_quote(text: str) -> bool:
    try:
        shlex.split(text, comments=True)
    except ValueError:
        return True
    return False


# --- checking them -----------------------------------------------------------

HIDDEN_FLAGS = {"--api-key"}  # accepted by the CLI but missing from every help page
HELP_FLAGS = {"-h", "--help"}
SHELL_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "&", "2>&1"}
PLACEHOLDER = re.compile(r"^(<.*>|\[.*\]|\{.*\}|\$.*|\.\.\.|…|[A-Z][A-Z0-9_/-]*)$")


def _known_flags(node: dict, inv: Inventory) -> set[str]:
    names: set[str] = set(HELP_FLAGS)
    for f in [*node.get("flags", []), *inv.global_flags]:
        names.add(f"--{f['name']}")
        if f.get("short"):
            names.add(f"-{f['short']}")
    return names


def check_command(text: str, inv: Inventory, where: str = "code") -> list[dict]:
    """Issues with one documented command. An empty list means it matches the CLI.

    Inline mentions (`where="inline"`) often just name a command, so they aren't
    checked for missing arguments.
    """
    try:
        argv = shlex.split(text, comments=True)
    except ValueError as e:
        return [{"kind": "unparseable", "detail": str(e)}]
    argv = argv[1:]  # drop "langsmith"
    for i, tok in enumerate(argv):
        if tok in SHELL_TOKENS or tok.startswith(("2>", ">")):
            argv = argv[:i]
            break
    if not argv:
        return []

    hidden = []
    for flag in HIDDEN_FLAGS:  # takes a value, but the inventory can't know that
        while flag in argv:
            i = argv.index(flag)
            hidden.append(flag)
            del argv[i : i + 2]
        hidden += [t.split("=", 1)[0] for t in argv if t.startswith(flag + "=")]
        argv = [t for t in argv if not t.startswith(flag + "=")]
    path_tokens, rest, flags = resolve(argv, inv)
    flags = hidden + flags
    path = " ".join(path_tokens)
    node = inv.get(path) or {}
    cmd = f"langsmith {path}".strip()
    issues: list[dict] = []

    known = _known_flags(node, inv)
    for flag in flags:
        if flag in known or flag == "--":
            continue
        if flag in HIDDEN_FLAGS:
            issues.append({"kind": "hidden_flag", "detail": f"`{flag}` works but no help page lists it"})
        else:
            issues.append({"kind": "unknown_flag", "detail": f"`{flag}` is not a flag of `{cmd}`"})
    if any(i["kind"] == "unknown_flag" for i in issues):
        return issues  # an unknown flag's value would be misread as a positional

    if rest and not node.get("runnable") and not PLACEHOLDER.match(rest[0]):
        issues.append({"kind": "unknown_command", "detail": f"`{rest[0]}` is not a subcommand of `{cmd}`"})
    elif where == "code" and node.get("runnable") and not (set(flags) & HELP_FLAGS):
        required = node.get("required_args", [])
        if len(rest) < len(required):
            issues.append({
                "kind": "missing_args",
                "detail": f"`{cmd}` requires {' '.join(required)}; the example passes {len(rest)} argument(s)",
            })
    return issues


def check_docs(snapshot_dir: Path, inv: Inventory) -> dict:
    snap = load_snapshot(snapshot_dir)
    results: list[dict] = []
    documented: set[str] = set()
    for page in snap["pages"]:
        if not check_page(page["url"]):
            continue
        text = (snapshot_dir / "pages" / page["file"]).read_text()
        seen: set[str] = set()
        for found in extract_commands(text):
            if found["text"] in seen:
                continue
            seen.add(found["text"])
            issues = check_command(found["text"], inv, found["where"])
            try:
                path_tokens, _, _ = resolve(shlex.split(found["text"], comments=True)[1:], inv)
                documented.add(" ".join(path_tokens))
            except ValueError:
                pass
            results.append({"url": page["url"], "title": page["title"], **found, "issues": issues})

    runnable = {c["path"] for c in inv.data["commands"] if c.get("runnable") and c["path"]}
    kinds = Counter(i["kind"] for r in results for i in r["issues"])
    return {
        "snapshot": str(snapshot_dir),
        "fetched_at": snap["fetched_at"],
        "cli_version": inv.data.get("cli_version"),
        "pages": len(snap["pages"]),
        "pages_with_commands": len({r["url"] for r in results}),
        "commands": len(results),
        "commands_with_issues": sum(1 for r in results if r["issues"]),
        "issue_counts": dict(kinds.most_common()),
        "runnable_documented": len(runnable & documented),
        "runnable_total": len(runnable),
        "undocumented": sorted(runnable - documented),
        "results": results,
    }


# --- search ------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
CHUNK_CHARS = 2500


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _chunks(markdown: str) -> list[tuple[str, str]]:
    """(heading, text) sections, split again if longer than CHUNK_CHARS."""
    out: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m and buf:
            out.append((heading, "\n".join(buf)))
            buf = []
        if m:
            heading = m.group(2).strip()
        buf.append(line)
    if buf:
        out.append((heading, "\n".join(buf)))
    split: list[tuple[str, str]] = []
    for h, text in out:
        for i in range(0, len(text), CHUNK_CHARS):
            split.append((h, text[i : i + CHUNK_CHARS]))
    return [(h, t) for h, t in split if t.strip()]


class DocsIndex:
    """BM25 keyword search over snapshot sections. Small and dependency-free on purpose."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.meta = load_snapshot(snapshot_dir)
        self.docs: list[dict] = []
        for page in self.meta["pages"]:
            text = (snapshot_dir / "pages" / page["file"]).read_text()
            for heading, chunk in _chunks(text):
                boost = _tokens(f"{page['title']} {heading}") * 2
                self.docs.append({
                    "url": page["url"], "title": page["title"], "heading": heading, "text": chunk,
                    "tf": Counter(_tokens(chunk) + boost),
                })
        for d in self.docs:
            d["len"] = sum(d["tf"].values())
        self.avg_len = sum(d["len"] for d in self.docs) / max(len(self.docs), 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d["tf"].keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def catalog(self) -> list[dict]:
        return [{"title": p["title"], "url": p["url"], "description": p["description"]} for p in self.meta["pages"]]

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[dict]:
        terms = [t for t in dict.fromkeys(_tokens(query)) if t in self.idf]
        scored = []
        for d in self.docs:
            s = 0.0
            for t in terms:
                f = d["tf"].get(t, 0)
                if f:
                    s += self.idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * d["len"] / self.avg_len))
            if s:
                scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        hits, per_page = [], Counter()
        for s, d in scored:
            if per_page[d["url"]] >= 2:  # keep results from more than one page
                continue
            per_page[d["url"]] += 1
            hits.append({"url": d["url"], "title": d["title"], "heading": d["heading"], "text": d["text"], "score": round(s, 2)})
            if len(hits) == k:
                break
        return hits
