# AGENTS.md — VAD engine adapter

**This file is identical in every `plugin-vad-*` repository.** It is copied verbatim from
`plugin-vad/templates/AGENTS.md`. Do not edit it in an engine repo — change the template
and re-copy it everywhere. Everything specific to *this* engine is in `ENGINE.md` beside
this file.

Assumes the workspace root `AGENTS.md` and the host framework rules in
`../plugin-vad/AGENTS.md`. Read both, then `ENGINE.md`.

---

## What this repository is

One engine adapter: a single class implementing the `VADEngine` Protocol from
`speechmux_plugin_vad.engine.base`, registered through a Python entry point.

The host framework (`plugin-vad`) owns the gRPC server, the per-session `StreamVAD` stream,
the `--config` YAML loader, the session-capacity check, `HealthCheck` and `GetCapabilities`.
**None of that lives here.** This repo has no `main.py`, no server code, and reads no config
file itself.

## Standard layout

```
plugin-vad-<impl>/
├── AGENTS.md                 # this file — verbatim copy of plugin-vad/templates/AGENTS.md
├── CLAUDE.md                 # one line: `@AGENTS.md` — never a copy of the rules
├── ENGINE.md                 # engine-specific rules, pitfalls and rationale
├── README.md                 # user-facing: install, config keys, model download
├── LICENSE
├── Makefile                  # identical across every engine repo
├── pyproject.toml            # dependency + entry point
├── src/speechmux_plugin_vad_<impl>/
│   ├── __init__.py
│   └── <impl>_engine.py      # the adapter class
└── tests/
```

Configuration lives in `plugin-vad/config/vad.yaml` under `engine.<engine_name>:`, because
that is where the process is launched from.

## Entry point

```toml
[project.entry-points."speechmux.vad_engine"]
<engine_name> = "speechmux_plugin_vad_<impl>.<impl>_engine:<ClassName>"
```

`<engine_name>` is snake_case and is what `server.engine:` in `vad.yaml` selects. The
package must be **installed** (`uv pip install -e .`) for the entry point to exist.

## Engine contract

```python
model_name: str
optimal_frame_ms: int
supported_sample_rates: list[int]
max_concurrent_sessions: int

def create_session_state(self, threshold: float) -> object
def process_frame(self, session_state, pcm_data: bytes, sample_rate: int) -> tuple[bool, float, float]
```

- Identity fields are **class attributes, not methods**; the host reads them directly to
  build `VADCapabilities`. `server.max_concurrent_sessions` in YAML may override the
  engine's default.
- **The engine is stateless with respect to sessions.** One instance serves every
  concurrent stream. All per-session state — model hidden state, frame buffers, the
  threshold — lives in the object returned by `create_session_state()` and is passed back
  on every `process_frame()` call. Two sessions must never share mutable state.
- `process_frame` returns `(is_speech, speech_probability, chunk_rms)`. It may be called
  with any chunk size Core sends; if the model needs fixed-size frames, buffer inside the
  session state and return the result for the frames completed by this chunk.
- `from_config(cls, config)` is the **only** way YAML reaches the engine. It receives the
  `engine.<engine_name>:` section as a dict. Never open a file yourself.
- Catch `ImportError` on the ML runtime and re-raise with an install hint.
- `sequence_number` echoing is done by the **host servicer**, not the engine. The engine
  never sees sequence numbers; it must simply return exactly one result per
  `process_frame` call, in order.

## `optimal_frame_ms` is advertised, not enforced

Core currently sends 30 ms frames regardless of what the engine advertises
(`../docs/plans/vad-frame-size-negotiation.md`). An engine that needs a different frame size
must re-buffer inside its session state. Keep `optimal_frame_ms` honest anyway — Core will
read it eventually.

## Build, test, lint

```bash
make install     # uv pip install -e ".[dev]" into ../.venv
make test        # pytest tests/ -v
make lint        # ruff check src/
make typecheck   # mypy src/
```

`Makefile` is identical across engine repos and defaults to `PYTHON ?= ../.venv/bin/python3`.
Do not diverge from it.

## Testing rules

- **Tests never load model weights.** Patch or mock the runtime so the suite runs on any
  machine. If the runtime itself must be importable for the tests to load, say so in
  `ENGINE.md` under "Test suite status" — that is a known cost, not a hidden one.
- Cover concurrent sessions explicitly: two session states processed interleaved must
  produce the same results as processed separately. That is the isolation guarantee.
- **Set every mock attribute the production code reads in a loop condition.** A bare
  `MagicMock()` is truthy and can hang a loop forever.

## Dependencies

`speechmux-plugin-vad` plus this engine's ML runtime (and `numpy` if needed). Nothing else.
Never depend on Core, another engine, or `grpc` server machinery.

## Do not

- Add a `main.py`, a gRPC server, or CLI flags — the host framework owns those.
- Read a config file directly; only `from_config` receives configuration.
- Share mutable model state across sessions.
- Return more or fewer than one result per `process_frame` call.
- Write a test that needs downloaded model weights.
- Diverge the `Makefile` from the other engine repos.
- Edit this file. Edit `plugin-vad/templates/AGENTS.md` and re-copy.
- Put anything in `CLAUDE.md` other than `@AGENTS.md`.

## Where the engine-specific rules are

`ENGINE.md` in this repo: declared capabilities, how per-session state is isolated, the
frame-size handling, config keys the engine reads, runtime quirks, pitfalls, and test-suite
status. Read it before changing the engine module.

## Related

- Host framework rules: `../plugin-vad/AGENTS.md`
- Engine contracts and lifecycle: `../docs/architecture/plugin-system.md`
- Wire contract: `../docs/api/plugin-protocol.md`
- Per-session VAD stream design: `../docs/decisions/0004-per-session-vad-stream.md`
- Adding another engine: `../.codex/skills/add-engine-plugin/SKILL.md`
