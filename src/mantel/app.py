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
import io
import json
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import config as cfgmod

MAX_TOOL_ROUNDS = 8
_STOP = object()  # sentinel: tell the host runner to stop

# mantel binds loopback, but loopback alone doesn't stop a malicious web page in
# the user's browser from POSTing to 127.0.0.1 (DNS-rebinding / CSRF). Since the
# server hosts MCP tools and mutates config, guard it: require a localhost Host
# (defeats rebinding) and reject any cross-origin Origin (defeats CSRF).
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


def _host_only(value: str) -> str:
    value = value.strip()
    if value.startswith("["):
        return value.split("]", 1)[0] + "]"
    return value.rsplit(":", 1)[0] if ":" in value else value


class _LocalGuard(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _host_only(request.headers.get("host", "")) not in _ALLOWED_HOSTS:
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and (urlparse(origin).hostname or "") not in _ALLOWED_HOSTS:
            return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
        return await call_next(request)


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


# ── attachments: server-side document → text extraction ───────────────────────
# Images are handled in the browser (read as base64 and sent inline as image_url
# parts straight to the vision model); this covers PDFs, DOCX, and text-ish files
# the browser can't parse itself.

_MAX_DOC_BYTES = 25 * 1024 * 1024   # reject uploads larger than this
_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # reject mic recordings larger than this
_MAX_DOC_CHARS = 200_000            # cap injected text so the prompt stays sane
_TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log",
    ".py", ".js", ".ts", ".tsx", ".html", ".css", ".scss", ".sh", ".c", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".java", ".rb", ".php", ".toml", ".ini", ".cfg",
    ".xml", ".sql", ".r", ".lua", ".pl", ".kt", ".swift",
}


class _ExtractError(Exception):
    pass


def _extract_text(name: str, data: bytes) -> str:
    """Best-effort text extraction from an uploaded document (PDF / DOCX / text)."""
    from pathlib import PurePosixPath

    ext = PurePosixPath(name).suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise _ExtractError("PDF support needs the 'pypdf' package") from e
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:
            raise _ExtractError("DOCX support needs the 'python-docx' package") from e
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    if ext in _TEXT_EXTS or not ext:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", "replace")
    raise _ExtractError(
        f"unsupported document type '{ext}' — attach an image, PDF, DOCX, or text file")


def _chats_db() -> sqlite3.Connection:
    """SQLite store for persisted conversations (~/.local/share/mantel/chats.db)."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    d = Path(base) / "mantel"
    d.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(d / "chats.db", timeout=10.0)
    con.execute("PRAGMA busy_timeout=5000")
    con.execute("CREATE TABLE IF NOT EXISTS conversations("
                "id TEXT PRIMARY KEY, title TEXT, created REAL, updated REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS messages("
                "conv_id TEXT, seq INTEGER, role TEXT, content TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id)")
    return con


def _has_image(messages: list) -> bool:
    """True if any message carries an image content part — those requests must go
    to a vision-capable model (a text model can't read images in the history)."""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            return True
    return False


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
            h = None
            try:
                h = _build_host(app.state.cfg)
                if h is not None:
                    await h.start()
            except Exception:  # initial start failed — reload can retry later
                h = None
            app.state.host = h
            ready.set()  # ALWAYS — never deadlock startup on a bad/slow server
            while True:
                item = await reload_q.get()
                app.state.host = None
                if h is not None:
                    await h.stop()
                if item is _STOP:
                    return
                try:
                    h = _build_host(app.state.cfg)
                    if h is not None:
                        await h.start()
                except Exception:
                    h = None
                app.state.host = h
                if not item.done():
                    item.set_result(len(h.openai_tools()) if h else 0)

        task = asyncio.create_task(runner())
        # Don't accept requests until the host is up — but never block forever
        # (per-server timeouts bound start(); this is a final backstop).
        try:
            await asyncio.wait_for(ready.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass
        try:
            yield
        finally:
            await reload_q.put(_STOP)
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    app = FastAPI(title="mantel", version="0.2.0", lifespan=lifespan)
    app.add_middleware(_LocalGuard)
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
                "providers": list(c.providers.keys()),
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

    # ── conversation persistence ─────────────────────────────────────────────
    @app.get("/api/chats")
    async def list_chats() -> JSONResponse:
        con = _chats_db()
        try:
            rows = con.execute(
                "SELECT c.id, c.title, c.updated, "
                "(SELECT count(*) FROM messages m WHERE m.conv_id = c.id) "
                "FROM conversations c ORDER BY c.updated DESC").fetchall()
        finally:
            con.close()
        return JSONResponse({"chats": [
            {"id": r[0], "title": r[1], "updated": r[2], "messages": r[3]} for r in rows]})

    @app.get("/api/chats/{cid}")
    async def get_chat(cid: str) -> JSONResponse:
        con = _chats_db()
        try:
            c = con.execute("SELECT id, title FROM conversations WHERE id=?", (cid,)).fetchone()
            if not c:
                return JSONResponse({"error": "not found"}, status_code=404)
            msgs = con.execute(
                "SELECT role, content FROM messages WHERE conv_id=? ORDER BY seq", (cid,)).fetchall()
        finally:
            con.close()
        out = []
        for role, content in msgs:
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                pass
            out.append({"role": role, "content": content})
        return JSONResponse({"id": c[0], "title": c[1], "messages": out})

    @app.put("/api/chats/{cid}")
    async def put_chat(cid: str, request: Request) -> JSONResponse:
        body = await request.json()
        title = (body.get("title") or "Untitled")[:120]
        messages = body.get("messages") or []
        now = time.time()
        con = _chats_db()
        try:
            exists = con.execute("SELECT 1 FROM conversations WHERE id=?", (cid,)).fetchone()
            if exists:
                con.execute("UPDATE conversations SET title=?, updated=? WHERE id=?", (title, now, cid))
            else:
                con.execute("INSERT INTO conversations(id,title,created,updated) VALUES(?,?,?,?)",
                            (cid, title, now, now))
            con.execute("DELETE FROM messages WHERE conv_id=?", (cid,))
            if not isinstance(messages, list):
                messages = []
            con.executemany(
                "INSERT INTO messages(conv_id,seq,role,content) VALUES(?,?,?,?)",
                [(cid, i, m.get("role", ""), json.dumps(m.get("content")))
                 for i, m in enumerate(messages) if isinstance(m, dict)])
            con.commit()
        finally:
            con.close()
        return JSONResponse({"ok": True, "id": cid, "title": title})

    @app.delete("/api/chats/{cid}")
    async def delete_chat(cid: str) -> JSONResponse:
        con = _chats_db()
        try:
            con.execute("DELETE FROM conversations WHERE id=?", (cid,))
            con.execute("DELETE FROM messages WHERE conv_id=?", (cid,))
            con.commit()
        finally:
            con.close()
        return JSONResponse({"ok": True})

    @app.post("/api/extract")
    async def api_extract(file: UploadFile = File(...)) -> JSONResponse:
        """Extract text from an uploaded document (PDF / DOCX / text) so the UI can
        include it in a chat message. Images are NOT sent here — the browser inlines
        them as image_url parts straight to the (vision) model."""
        # Read at most the cap (+1 to detect overflow) so a giant upload can't
        # balloon memory (Starlette has already spooled the body to a temp file).
        data = await file.read(_MAX_DOC_BYTES + 1)
        if len(data) > _MAX_DOC_BYTES:
            return JSONResponse({"error": "file too large (max 25 MB)"}, status_code=413)
        try:
            text = _extract_text(file.filename or "file", data)
        except _ExtractError as e:
            return JSONResponse({"error": str(e)}, status_code=415)
        except Exception as e:  # never 500 on a malformed upload
            return JSONResponse({"error": f"could not read document: {e}"}, status_code=422)
        truncated = len(text) > _MAX_DOC_CHARS
        return JSONResponse({"name": file.filename, "chars": min(len(text), _MAX_DOC_CHARS),
                             "truncated": truncated, "text": text[:_MAX_DOC_CHARS]})

    @app.post("/api/transcribe")
    async def api_transcribe(file: UploadFile = File(...)) -> JSONResponse:
        """Speech-to-text: proxy recorded audio to the backend's OpenAI-style
        /audio/transcriptions (hearth's faster-whisper route) and return the text."""
        data = await file.read(_MAX_AUDIO_BYTES + 1)
        if len(data) > _MAX_AUDIO_BYTES:
            return JSONResponse({"error": "audio too large (max 25 MB)"}, status_code=413)
        prov = app.state.cfg.active()
        files = {"file": (file.filename or "speech.webm", data, file.content_type or "audio/webm")}
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                r = await client.post(f"{prov.base_url}/audio/transcriptions",
                                      files=files, data={"model": "whisper-1"},
                                      headers=prov.headers())
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"STT backend unreachable: {e}"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": r.text[:300] or "transcription failed"},
                                status_code=r.status_code)
        try:
            return JSONResponse({"text": (r.json() or {}).get("text", "")})
        except ValueError:
            return JSONResponse({"text": r.text})

    @app.post("/api/speak")
    async def api_speak(request: Request):
        """Text-to-speech: proxy to the backend's OpenAI-style /audio/speech
        (hearth's Piper route, or a cloud TTS) and return the audio bytes."""
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            return JSONResponse({"error": "missing 'text'"}, status_code=400)
        prov = app.state.cfg.active()
        req = {"model": "tts-1", "input": text[:4000]}  # model ignored by hearth; valid for OpenAI
        voice = body.get("voice") or app.state.cfg.tts_voice
        if voice:
            req["voice"] = voice
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(f"{prov.base_url}/audio/speech", json=req, headers=prov.headers())
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"TTS backend unreachable: {e}"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": r.text[:300] or "TTS failed"}, status_code=r.status_code)
        return Response(content=r.content,
                        media_type=r.headers.get("content-type", "audio/wav"))

    @app.post("/api/chat")
    async def api_chat(request: Request) -> StreamingResponse:
        body = await request.json()
        prov = app.state.cfg.active()
        host = app.state.host
        messages = list(body.get("messages") or [])
        base = {k: v for k, v in body.items() if k not in ("messages", "stream")}
        # An image anywhere in the conversation forces a vision-capable model — a
        # text-only chat model can't read image_url parts. Route to the configured
        # vision model (hearth's `vision` role by default). Only for LOCAL providers
        # (no api_key): a cloud provider has no "vision" role, and its selected model
        # (e.g. gpt-4o) already handles images, so leave that request untouched.
        vmodel = app.state.cfg.vision_model
        if vmodel and not prov.api_key and _has_image(messages):
            base = {**base, "model": vmodel}
            # Force the OpenAI passthrough (which carries image_url parts) — a
            # `think` flag would route hearth to its native path, which doesn't
            # translate images. Vision doesn't need the deep-think toggle anyway.
            base.pop("think", None)
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
                        h = app.state.host  # read fresh — a concurrent reload may have swapped it
                        res = await h.call(name, args) if h else {"ok": False, "content": "MCP host unavailable"}
                        yield _tool_frame("result", cid, name, res)  # full result (inline images) → UI
                        tool_content = res.get("content", "")
                        # A generated image returns as a ~2 MB data: URL — render it for the
                        # user (frame above) but feed the model only a short placeholder so it
                        # doesn't blow up the context (and the model can't read base64 anyway).
                        if isinstance(tool_content, str) and tool_content.startswith("data:image/"):
                            tool_content = "[image generated and shown to the user]"
                        messages.append({"role": "tool", "tool_call_id": cid, "content": tool_content})
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
