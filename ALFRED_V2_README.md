# ALFRED v2 — Web UI, Persona System & Dual-Model Support

## What's new in this commit

ALFRED moved from a CLI-only tool to a full client-server application, with a persistent identity and a web-based chat interface.

### Architecture changes

- **`agent.py`** — extracted all shared agent-building logic (LLM setup, system prompt construction, tool config, SFT logging, output-saving) into a single module reused by both the CLI (`main.py`) and the FastAPI server (`server.py`), so neither has duplicated logic.
- **`server.py`** — new FastAPI backend exposing:
  - `POST /session/new` — starts a conversation, fixing the model choice (`local` or `openrouter`) for that thread
  - `POST /chat` — sends a message to the agent bound to a given `thread_id`
  - `GET /history/{thread_id}` — replays a conversation's prior turns
  - Serves the static frontend from the same app (no CORS needed)
- **`static/index.html`** — single-file chat UI. Model choice happens once at session start; `thread_id` persists in `localStorage` so a page refresh resumes the same conversation.

### Persona system

Previously, ALFRED had no persistent identity — asked "who are you," it would default to describing itself as a generic AI assistant. Identity now lives in **`PERSONA.md`**, loaded once at startup and composed with the existing operational rules to form the full system prompt:

```python
SYSTEM_PROMPT = f"{PERSONA}\n\n{OPERATIONAL_RULES}".strip()
```

Editing `PERSONA.md` changes ALFRED's voice without touching any tool logic or output-formatting rules — the two concerns are fully separated.

### Dual-model support, session-scoped

Each session is bound to a single model (`local` Ollama/Qwen2.5:14b, or `openrouter`/`poolside/laguna-s-2.1:free`) at creation time via `/session/new`, rather than per-message. Both agents are built **once**, at server startup, and share a single `SqliteSaver` checkpointer instance — safe for FastAPI's threadpool because `SqliteSaver` internally serializes access through its own lock, so concurrent requests never corrupt shared conversation state.

### Known follow-ups (not yet done)

- `search_tool` in `tools.py` has no request timeout — a slow DuckDuckGo query can hang a request for several minutes. Needs a bounded timeout with a graceful fallback message.
- No auth on `/history/{thread_id}` — thread IDs are stored client-side in `localStorage` and aren't validated server-side beyond existence. Fine for local single-user use; would need addressing before any non-local deployment.
- `birthday()`-style state mutation bugs aside (unrelated to this app — just a reminder to self to always double check in-place vs. reassignment when updating object state).

## Running it

```bash
source venv/bin/activate
uvicorn server:app --reload
```
Open `http://127.0.0.1:8000/`.

CLI still works unchanged:
```bash
python main.py --local   # or omit --local for OpenRouter
```
