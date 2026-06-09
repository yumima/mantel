"""Mantel's local app server.

Serves the chat UI at ``/`` and proxies chat/models to the configured backend
under ``/api/*``. The UI is **same-origin** with mantel, so the browser/webview
streams from mantel directly; mantel talks to the backend **server-to-server**,
which sidesteps cross-origin blocks (hearth, e.g., is loopback-only with no
CORS) and is exactly where the MCP host + tool-call loop land in M2.

M0: a transparent streaming proxy of the OpenAI ``/chat/completions`` and
``/models`` shapes to the single active provider (default: local hearth).
"""

from __future__ import annotations

import json
from importlib import resources

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import config as cfgmod


def _index_html() -> str | None:
    try:
        return resources.files("mantel").joinpath("webui/index.html").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None


def create_app(cfg: cfgmod.Config | None = None) -> FastAPI:
    cfg = cfg or cfgmod.load()
    app = FastAPI(title="mantel", version="0.1.0")
    app.state.cfg = cfg

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        html = _index_html()
        if html is None:
            return HTMLResponse("<h1>Mantel</h1><p>UI asset missing from this install.</p>",
                                status_code=500)
        return HTMLResponse(html)

    @app.get("/api/health", include_in_schema=False)
    async def health() -> dict:
        return {"ok": True, "app": "mantel", "version": "0.1.0"}

    @app.get("/api/config")
    async def api_config() -> dict:
        c = app.state.cfg
        return {"provider": c.provider, "model": c.model, "backend": c.active().base_url}

    @app.get("/api/models")
    async def api_models() -> JSONResponse:
        prov = app.state.cfg.active()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{prov.base_url}/models", headers=prov.headers())
            return JSONResponse(r.json(), status_code=r.status_code)
        except (httpx.HTTPError, ValueError) as e:
            # Backend down / unreachable — the UI shows "no engine", not a crash.
            return JSONResponse({"error": str(e), "data": []}, status_code=502)

    @app.post("/api/chat")
    async def api_chat(request: Request) -> StreamingResponse:
        body = await request.json()
        body["stream"] = True
        prov = app.state.cfg.active()
        headers = {"Content-Type": "application/json", **prov.headers()}

        async def gen():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{prov.base_url}/chat/completions",
                                             json=body, headers=headers) as r:
                        if r.status_code != 200:
                            detail = (await r.aread()).decode("utf-8", "ignore")[:300]
                            yield _err_frame(f"backend {r.status_code}: {detail}")
                            return
                        async for chunk in r.aiter_bytes():
                            yield chunk
            except httpx.HTTPError as e:
                yield _err_frame(f"backend unreachable ({prov.base_url}): {e}")

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _err_frame(message: str) -> bytes:
    """An SSE error frame the UI understands (mirrors the OpenAI error shape)."""
    return f"data: {json.dumps({'error': {'message': message}})}\n\n".encode("utf-8")
