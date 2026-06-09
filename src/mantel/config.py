"""Mantel configuration — providers + the active model.

M0 scope: a single default provider (local **hearth**, OpenAI-compatible on
loopback) and the active model. Stored at ``~/.config/mantel/config.yaml``;
sensible defaults apply when it's absent, so a fresh install Just Works against
a running hearth with zero config.

Forward shape (M1+): multiple providers (OpenAI/Anthropic/…), profiles that
bundle ``provider + model + mcp servers + system prompt``, and an ``mcp_servers``
section. Kept deliberately small here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# hearth's OpenAI-compatible gateway on loopback (see github.com/yumima/hearth).
DEFAULT_HEARTH_URL = "http://127.0.0.1:11435/v1"


@dataclass
class Provider:
    """An OpenAI-compatible backend. ``api_key`` is empty for local hearth."""
    type: str = "openai"          # only "openai" (compatible) in M0
    base_url: str = DEFAULT_HEARTH_URL
    api_key: str = ""

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


@dataclass
class Config:
    host: str = "127.0.0.1"       # mantel's own local server (loopback only)
    port: int = 8765
    provider: str = "hearth"      # the active provider key
    model: str = "primary_chat"   # hearth role alias by default
    providers: dict[str, Provider] = field(default_factory=lambda: {"hearth": Provider()})

    def active(self) -> Provider:
        return self.providers.get(self.provider, Provider())


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "mantel" / "config.yaml"


def load() -> Config:
    """Load config, falling back to all-defaults (hearth on :11435) when the file
    is absent or unreadable — mantel must never fail to start over config."""
    cfg = Config()
    p = config_path()
    if not p.exists():
        return cfg
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return cfg
    cfg.host = data.get("host", cfg.host)
    try:
        cfg.port = int(data.get("port", cfg.port))
    except (TypeError, ValueError):
        pass
    cfg.provider = data.get("provider", cfg.provider)
    cfg.model = data.get("model", cfg.model)
    provs = data.get("providers") or {}
    if isinstance(provs, dict) and provs:
        cfg.providers = {}
        for name, v in provs.items():
            v = v or {}
            cfg.providers[name] = Provider(
                type=v.get("type", "openai"),
                base_url=v.get("base_url", DEFAULT_HEARTH_URL),
                api_key=v.get("api_key", ""),
            )
    cfg.providers.setdefault(cfg.provider, Provider())  # active must resolve
    return cfg
