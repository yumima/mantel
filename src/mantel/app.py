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

import asyncio
import json
from contextlib import asynccontextmanager
from importlib import resources

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import config as cfgmod

MAX_TOOL_ROUNDS = 8
_STOP = object()  # sentinel: tell the host runner to stop


async def _reload(app: FastAPI) -> int | None:
    """Ask the host runner to rebuild from the (already-mutated) config; returns
    the new tool count, or None if there's no runner (e.g. a test-injected host)."""
    rq = getattr(app.state, "reload", None)
    if rq is None:
        return None
    fut = asyncio.get_running_loop().create_future()
    await rq.put(fut)
    return await fut


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
        app.state.cfg = cfg
        if external_host:
            # Tests inject a pre-started host and manage its lifecycle themselves.
            app.state.host = host
            app.state.reload = None
            yield
            return

        # The host's MCP sessions must be started, reloaded, and stopped in ONE
        # long-lived task (anyio task-group contexts can't cross tasks). So a
        # runner task owns them: it starts the host, then services reload/stop
        # requests off a queue. Endpoints trigger reloads via _reload().
        ready = asyncio.Event()
        reload_q: asyncio.Queue = asyncio.Queue()
        app.state.reload = reload_q

        async def runner():
            h = _build_host(app.state.cfg)
            if h is not None:
                await h.start()
            app.state.host = h
            ready.set()
            while True:
                item = await reload_q.get()
                app.state.host = None
                if h is not None:
                    await h.stop()
                if item is _STOP:
                    return
                h = _build_host(app.state.cfg)
                if h is not None:
                    await h.start()
                app.state.host = h
                if not item.done():
                    item.set_result(len(h.openai_tools()) if h else 0)

        task = asyncio.create_task(runner())
        await ready.wait()  # don't accept requests until the initial host is up
        try:
            yield
        finally:
            await reload_q.put(_STOP)
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

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

    @app.get("/api/servers")
    async def api_servers() -> dict:
        c = app.state.cfg
        h = app.state.host
        return {"servers": [{"name": n, "enabled": s.enabled, "command": s.command,
                             "args": s.args, "auto_approve": s.auto_approve}
                            for n, s in c.mcp_servers.items()],
                "errors": h.errors if h else {}}

    @app.post("/api/mcp/{name}/{action}")
    async def api_mcp(name: str, action: str) -> JSONResponse:
        c = app.state.cfg
        if name not in c.mcp_servers:
            return JSONResponse({"error": f"no such server '{name}'"}, status_code=404)
        if action not in ("enable", "disable"):
            return JSONResponse({"error": "action must be enable or disable"}, status_code=400)
        c.mcp_servers[name].enabled = (action == "enable")
        cfgmod.save(c)                       # persist across restarts
        tools = await _reload(app)           # apply live
        h = app.state.host
        return JSONResponse({"ok": True, "server": name, "enabled": c.mcp_servers[name].enabled,
                             "tools": tools if tools is not None else (len(h.openai_tools()) if h else 0),
                             "errors": h.errors if h else {}})

    @app.post("/api/provider/{name}")
    async def api_provider(name: str) -> JSONResponse:
        c = app.state.cfg
        if name not in c.providers:
            return JSONResponse({"error": f"no such provider '{name}'"}, status_code=404)
        c.provider = name                    # the chat loop reads active() per request → live
        cfgmod.save(c)
        return JSONResponse({"ok": True, "provider": name, "backend": c.active().base_url})

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
