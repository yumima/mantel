"""Web fetch — a local MCP server giving the agent read access to web pages.

``fetch(url)`` GETs an http(s) URL and returns its readable text (HTML tags
stripped, script/style dropped). No API key, no search provider — just retrieval
of pages/APIs the agent (on the user's behalf) asks for. The user enables it from
the Tools & Settings panel; tool calls remain deny-by-default in the host.

Run standalone over stdio:  python -m mantel.mcp_servers.web
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx
from mcp.server.fastmcp import FastMCP

MAX_CHARS = 20000
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def _html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)  # last-resort tag strip
    return re.sub(r"\n{3,}", "\n\n", "\n".join(p.parts)).strip()


def _do_fetch(url: str, max_chars: int = MAX_CHARS) -> dict:
    if not re.match(r"^https?://", url, re.I):
        return {"error": "only http(s) URLs are allowed"}
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True,
                          headers={"User-Agent": "mantel-fetch/0.1"}) as client:
            r = client.get(url)
    except httpx.HTTPError as e:
        return {"error": f"fetch failed: {e}"}
    ctype = r.headers.get("content-type", "")
    text = _html_to_text(r.text) if "html" in ctype.lower() else r.text.strip()
    return {"status": r.status_code, "url": str(r.url), "content_type": ctype,
            "text": text[: max(1, max_chars)], "truncated": len(text) > max_chars}


mcp = FastMCP("web")


@mcp.tool()
def fetch(url: str, max_chars: int = MAX_CHARS) -> str:
    """Fetch a web page or HTTP API by URL and return its readable text (HTML tags
    stripped). Use to read documentation, articles, or web data. http/https only."""
    r = _do_fetch(url, max_chars)
    if "error" in r:
        return r["error"]
    head = f"[{r['status']}] {r['url']} ({r['content_type']})\n\n"
    return head + r["text"] + ("\n\n…(truncated)" if r["truncated"] else "")


if __name__ == "__main__":
    mcp.run()
