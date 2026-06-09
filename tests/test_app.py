"""Mantel server tests — UI route + the /api proxy, against an UNREACHABLE
backend so they're deterministic and fast (no real engine needed)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mantel import config as cfgmod
from mantel.app import create_app


def _client() -> TestClient:
    cfg = cfgmod.Config()
    # Point the backend at a closed port so proxy calls fail fast (refused),
    # exercising mantel's graceful-degradation paths without a running engine.
    cfg.provider = "hearth"
    cfg.providers = {"hearth": cfgmod.Provider(base_url="http://127.0.0.1:1/v1")}
    return TestClient(create_app(cfg))


def test_index_serves_ui():
    r = _client().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<title>Mantel</title>" in r.text
    assert "/api/chat" in r.text                 # same-origin wiring
    assert 'src="http' not in r.text and "<link" not in r.text  # self-contained


def test_health():
    r = _client().get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_config_exposes_active_backend():
    body = _client().get("/api/config").json()
    assert body["provider"] == "hearth"
    assert body["model"]
    assert body["backend"].startswith("http://")


def test_models_degrades_gracefully_when_backend_down():
    r = _client().get("/api/models")
    assert r.status_code == 502                  # backend unreachable…
    assert r.json().get("data") == []            # …but a usable shape, no crash


def test_chat_streams_an_error_frame_when_backend_down():
    with _client().stream("POST", "/api/chat",
                          json={"model": "primary_chat", "messages": [{"role": "user", "content": "hi"}]}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = b"".join(r.iter_bytes())
    assert b"error" in body and b"unreachable" in body  # surfaced, not swallowed
