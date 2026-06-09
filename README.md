# mantel

**A universal, MCP-native desktop chat client — local-first, any backend.**

Mantel is a clickable chat app that talks to **any** model backend and defaults
to a **local [hearth](https://github.com/yumima/hearth) engine** on loopback. It
(will) host **MCP servers** so the model can use *your* tools — files, search,
databases, your own services — with **you** deciding which tools, regardless of
which model you run. Your data and tools stay local unless you explicitly opt
into a cloud backend or a remote server.

> **Status: early (M0)** — a working chat window against a local hearth backend.
> The MCP host, multi-provider support, and profiles are on the roadmap below.

## Why (vs. ChatGPT / Claude desktop)
- **No vendor lock-in.** One UI, any model — a free local model for most work, a
  frontier cloud model when you need it. The big clients are single-vendor.
- **Your tools, via MCP.** The tools the model can use are *yours* to configure —
  not a vendor's curated connector list — and work with whatever model you pick.
- **Local-first & private.** Defaults are local, no keys, no egress; cloud and
  remote MCP servers are explicit, labeled opt-ins.
- **Profiles** *(roadmap)* — one click swaps the whole stack: backend + model +
  MCP servers + system prompt.

## Quick start
```bash
git clone https://github.com/yumima/mantel && cd mantel
python3 -m venv .venv && .venv/bin/pip install -e ".[gui]"   # [gui] = native window (pywebview)

# with a hearth engine running on :11435 (see github.com/yumima/hearth):
.venv/bin/mantel            # opens the desktop chat window
# or:
.venv/bin/mantel serve      # headless — open http://127.0.0.1:8765 in a browser
.venv/bin/mantel install    # add a clickable launcher to your app menu
```
Without `[gui]`/pywebview, `mantel` falls back to a Chromium/Edge `--app` window,
then your default browser.

## Commands
| Command | What it does |
|---|---|
| `mantel` | open the desktop chat window (starts mantel's local server) |
| `mantel serve` | run the local server in the foreground (headless / dev) |
| `mantel install` / `uninstall` | add / remove the clickable desktop launcher |

## Architecture
Mantel runs a tiny **local server** (FastAPI) that serves the chat UI and proxies
chat/models to the configured backend under `/api/*`. The UI is **same-origin**
with mantel, so streaming `fetch` works with no CORS; mantel talks to the backend
**server-to-server** (so a loopback-only engine like hearth, with no CORS, works
fine). That proxy is exactly where the **MCP host + tool-call loop** land (M2).

```
window (pywebview)  →  mantel server (FastAPI)  →  backend (hearth / OpenAI / …)
   UI at  /               /api/chat  /api/models      /v1/chat/completions
                          + MCP host (M2) ───────────→ MCP servers (your tools)
```

## Config
`~/.config/mantel/config.yaml` (optional — absent ⇒ local hearth defaults):
```yaml
host: 127.0.0.1
port: 8765
provider: hearth
model: primary_chat
providers:
  hearth: { type: openai, base_url: http://127.0.0.1:11435/v1 }
```

## Roadmap
- **M0 — scaffold + chat-to-hearth** ✅ *(this)*
- **M1 — multi-backend** — provider registry + key storage; OpenAI / Anthropic / Gemini; model picker.
- **M2 — MCP host** — spawn stdio/SSE servers, aggregate tools, the agent tool-loop, an approval UX; default servers (filesystem / memory / time / git / fetch).
- **M3 — profiles + settings UI** — profile switcher, server catalog, keys & scopes.
- **M4 — connectors + packs** — credentialed servers (search / github / …), remote servers, domain packs.
- **M5 — distribution** — standalone bundles + GitHub releases.

## Develop
```bash
pip install -e ".[dev]"
pytest -q
```

## License
[Apache-2.0](LICENSE).
