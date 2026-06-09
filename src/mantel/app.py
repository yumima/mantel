"""Mantel's local app server.

Serves the chat UI at ``/`` and, under ``/api/*``, runs the **agent loop**: it
calls the configured backend with the aggregated **MCP** tools, executes any
tool calls the model makes (deny-by-default, via the MCP host), feeds the
results back, and loops until the model answers — streaming the answer and the
tool activity to the UI.

The UI is **same-origin** with mantel, so streaming ``fetch`` works with no
CORS; mantel talks to the backend **server-to-server** (so a loopback-only
engine like hearth, with no CORS, works fine). When no MCP servers are
configured the loop runs once → it's a plain streaming chat.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from importlib import resources

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import config as cfgmod

MAX_TOOL_ROUNDS = 8


class _BackendError(Exception):
    pass


def _index_html() -> str | None:
    try:
        return resources.files("mantel").joinpath("webui/index.html").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


def _build_host(cfg: cfgmod.Config):
    if not cfg.mcp_servers:
        return None
    from .mcp_host import MCPHost
    return MCPHost(cfg.mcp_servers)


def create_app(cfg: cfgmod.Config | None = None, host=None) -> FastAPI:
    cfg = cfg or cfgmod.load()
    external_host = host is not None  # tests pass a pre-started host; we don't manage it

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        h = host if external_host else _build_host(cfg)
        if h is not None and not external_host:
            await h.start()
        app.state.host = h
        try:
            yield
        finally:
            if h is not None and not external_host:
                await h.stop()

    app = FastAPI(title="mantel", version="0.2.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.host = None

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        html = _index_html()
        if html is None:
            return HTMLResponse("<h1>Mantel</h1><p>UI asset missing from this install.</p>",
                                status_code=500)
        return HTMLResponse(html)

    @app.get("/api/health", include_in_schema=False)
    async def health() -> dict:
        h = app.state.host
        n = len(h.openai_tools()) if h else 0
        return {"ok": True, "app": "mantel", "version": "0.2.0", "tools": n}

    @app.get("/api/config")
    async def api_config() -> dict:
        c = app.state.cfg
        h = app.state.host
        return {"provider": c.provider, "model": c.model, "backend": c.active().base_url,
                "tools": len(h.openai_tools()) if h else 0}

    @app.get("/api/tools")
    async def api_tools() -> dict:
        h = app.state.host
        if h is None:
            return {"tools": [], "errors": {}}
        return {
            "tools": [{"name": t["function"]["name"], "description": t["function"]["description"]}
                      for t in h.openai_tools()],
            "errors": h.errors,
        }

    @app.get("/api/models")
    async def api_models() -> JSONResponse:
        prov = app.state.cfg.active()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{prov.base_url}/models", headers=prov.headers())
            return JSONResponse(r.json(), status_code=r.status_code)
        except (httpx.HTTPError, ValueError) as e:
            return JSONResponse({"error": str(e), "data": []}, status_code=502)

    @app.post("/api/chat")
    async def api_chat(request: Request) -> StreamingResponse:
        body = await request.json()
        prov = app.state.cfg.active()
        host = app.state.host
        messages = list(body.get("messages") or [])
        base = {k: v for k, v in body.items() if k not in ("messages", "stream")}
        tools = host.openai_tools() if (host and host.has_tools()) else None

        async def gen():
            try:
                for _round in range(MAX_TOOL_ROUNDS):
                    req = {**base, "messages": messages, "stream": True}
                    if tools:
                        req["tools"] = tools
                    content = ""
                    tcs: dict[int, dict] = {}
                    try:
                        async for payload in _backend_chat_stream(prov, req):
                            try:
                                obj = json.loads(payload)
                            except ValueError:
                                continue
                            if obj.get("error"):
                                yield _err_frame((obj["error"] or {}).get("message", "backend error"))
                                return
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            if delta.get("content"):
                                content += delta["content"]
                                yield _content_frame(delta["content"])
                            reason = delta.get("reasoning") or delta.get("reasoning_content")
                            if reason:
                                yield _reasoning_frame(reason)
                            for tc in (delta.get("tool_calls") or []):
                                _accumulate_tool_call(tcs, tc)
                    except _BackendError as e:
                        yield _err_frame(str(e))
                        return
                    except httpx.HTTPError as e:
                        yield _err_frame(f"backend unreachable ({prov.base_url}): {e}")
                        return

                    calls = _finalize_tool_calls(tcs)
                    if not calls:
                        break  # the final answer was streamed above

                    messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    for call in calls:
                        name = (call.get("function") or {}).get("name", "")
                        try:
                            args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        cid = call.get("id", "")
                        yield _tool_frame("call", cid, name, {"arguments": args})
                        res = await host.call(name, args) if host else {"ok": False, "content": "no MCP host"}
                        yield _tool_frame("result", cid, name, res)
                        messages.append({"role": "tool", "tool_call_id": cid,
                                         "content": res.get("content", "")})
                else:
                    yield _tool_frame("note", "", "", {"content": f"stopped after {MAX_TOOL_ROUNDS} tool rounds"})
                yield b"data: [DONE]\n\n"
            except Exception as e:  # never crash the stream
                yield _err_frame(f"mantel error: {e}")
                yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# ── backend streaming (isolated so tests can fake the backend) ────────────────


async def _backend_chat_stream(prov: cfgmod.Provider, req: dict):
    """Yield each backend SSE ``data:`` payload (prefix stripped, ``[DONE]``
    excluded). Raises ``_BackendError`` on a non-200 response."""
    headers = {"Content-Type": "application/json", **prov.headers()}
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{prov.base_url}/chat/completions",
                                 json=req, headers=headers) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode("utf-8", "ignore")[:300]
                raise _BackendError(f"backend {r.status_code}: {detail}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                yield payload


# ── SSE frame builders + streamed-tool-call accumulation ──────────────────────


def _content_frame(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n".encode("utf-8")


def _reasoning_frame(text: str) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': {'reasoning': text}}]})}\n\n".encode("utf-8")


def _tool_frame(phase: str, cid: str, name: str, payload: dict) -> bytes:
    evt = {"mantel_tool": {"phase": phase, "id": cid, "name": name, **payload}}
    return f"data: {json.dumps(evt)}\n\n".encode("utf-8")


def _err_frame(message: str) -> bytes:
    return f"data: {json.dumps({'error': {'message': message}})}\n\n".encode("utf-8")


def _accumulate_tool_call(tcs: dict[int, dict], tc: dict) -> None:
    """Merge a streamed tool_call delta into the per-index accumulator (OpenAI
    streams the id, name, and arguments across multiple deltas)."""
    i = tc.get("index", 0)
    slot = tcs.setdefault(i, {"id": "", "type": "function",
                              "function": {"name": "", "arguments": ""}})
    if tc.get("id"):
        slot["id"] = tc["id"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        slot["function"]["name"] = fn["name"]
    if fn.get("arguments"):
        slot["function"]["arguments"] += fn["arguments"]


def _finalize_tool_calls(tcs: dict[int, dict]) -> list[dict]:
    return [tcs[i] for i in sorted(tcs) if tcs[i]["function"]["name"]]
