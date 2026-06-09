"""mantel CLI.

  mantel            open the desktop chat window (starts mantel's local server)
  mantel serve      run the local server in the foreground (headless / dev)
  mantel install    add a clickable desktop launcher (app menu / Start menu)
  mantel uninstall  remove the launcher

mantel is a *client*: bare `mantel` opens the window. The window runs mantel's
own loopback server (a daemon thread) which serves the UI and proxies to the
configured backend (default: local hearth on :11435).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import httpx

from . import config as cfgmod


def _server_url(cfg: cfgmod.Config) -> str:
    return f"http://{cfg.host}:{cfg.port}/"


def _server_up(cfg: cfgmod.Config, timeout: float = 0.5) -> bool:
    try:
        return httpx.get(f"http://{cfg.host}:{cfg.port}/api/health",
                         timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    cfg = cfgmod.load()
    print(f"[mantel] serving http://{cfg.host}:{cfg.port}  (backend: {cfg.active().base_url})")
    print(f"[mantel] open the UI at {_server_url(cfg)}")
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port,
                log_level=getattr(args, "log_level", "info"))
    return 0


def _start_server_thread(cfg: cfgmod.Config) -> bool:
    """Start mantel's server in a daemon thread (so it dies with the window).
    Returns True once it's serving — or was already up (e.g. a `mantel serve`)."""
    if _server_up(cfg):
        return True
    import uvicorn

    from .app import create_app

    server = uvicorn.Server(uvicorn.Config(
        create_app(cfg), host=cfg.host, port=cfg.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if _server_up(cfg):
            return True
        time.sleep(0.1)
    return False


def cmd_open(args: argparse.Namespace) -> int:
    cfg = cfgmod.load()
    if not _start_server_thread(cfg):
        print("warning: mantel's local server didn't come up; the window may be blank.",
              file=sys.stderr)
    from . import desktop
    return desktop.open_window(_server_url(cfg))


def cmd_install(args: argparse.Namespace) -> int:
    from . import desktop
    return desktop.install(_mantel_exec_argv())


def cmd_uninstall(args: argparse.Namespace) -> int:
    from . import desktop
    return desktop.uninstall()


def _mantel_exec_argv() -> list[str]:
    """argv a launcher uses to invoke mantel — bare `mantel` opens the window.
    Prefers a stable ~/.local/bin/mantel symlink (so venv rebuilds don't break
    the launcher); falls back to `<python> -m mantel`."""
    real = shutil.which("mantel")
    if real:
        return [_stable_link(os.path.realpath(real))]
    return [os.path.realpath(sys.executable), "-m", "mantel"]


def _stable_link(real: str) -> str:
    link = Path.home() / ".local" / "bin" / "mantel"
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() and not link.is_symlink():
            return str(link)  # a real file already sits there — trust it
        if not link.exists() or os.path.realpath(link) != real:
            if link.is_symlink():
                link.unlink()
            link.symlink_to(real)
        return str(link)
    except OSError:
        return real


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mantel",
        description="Universal, MCP-native chat client — defaults to a local hearth engine.")
    p.set_defaults(func=cmd_open)  # bare `mantel` opens the window
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("serve", help="run the local server in the foreground (headless / dev)")
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    sub.add_parser("open", help="open the desktop chat window").set_defaults(func=cmd_open)
    sub.add_parser("install", help="install the clickable desktop app (app menu / Start menu)") \
       .set_defaults(func=cmd_install)
    sub.add_parser("uninstall", help="remove the desktop app launcher") \
       .set_defaults(func=cmd_uninstall)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
