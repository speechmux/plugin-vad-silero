# ENGINE.md — Silero VAD

Engine-specific rules for `plugin-vad-silero`. Common rules for every VAD engine adapter are
in `AGENTS.md` beside this file.

---

## Role

Wraps the Silero VAD model (via `silero-vad` + `torch`). The production VAD for SpeechMux
and currently the only real one; `plugin-vad` ships a `dummy` for load testing.

## Entry point and names

| Name | Value |
|------|-------|
| Entry-point name (`server.engine:`) | `silero` |
| Class | `speechmux_plugin_vad_silero.silero_engine:SileroVADEngine` |
| Config section | `engine.silero:` in `plugin-vad/config/vad.yaml` |

## Declared capabilities

```python
model_name = "silero_vad"
optimal_frame_ms = 32                    # 512 samples @ 16 kHz — Silero's required frame
supported_sample_rates = [8000, 16000]   # set in __init__
max_concurrent_sessions = 50             # overridable via server.max_concurrent_sessions
```

## Session state and isolation

Silero is an RNN with hidden state, so two sessions cannot share one model instance.
`create_session_state()` returns a `_SileroSessionState` holding a **private model copy**.
`deepcopy` is tried first; some TorchScript builds are non-picklable, so there is a fallback
that reloads the model from scratch. Do not "optimise" this into a shared model —
cross-session state bleed is silent and produces plausible-looking wrong boundaries.
`tests/test_concurrent_streams.py` guards this.

## Frame handling

Silero needs exactly 512 samples. Core sends 30 ms frames
(`../docs/plans/vad-frame-size-negotiation.md`), so incoming audio is accumulated in a deque
inside the session state, tracking `buffered_samples` and `chunk_offset`, until a full frame
exists. `process_frame` scores every complete frame in the chunk and returns the **maximum**
probability across them.

`_MAX_BUFFERED_SAMPLES` discards the buffer past 30 s of audio. Core never sends chunks that
large; the cap guards against a misbehaving client. Removing it turns a client bug into
unbounded memory growth.

## Configuration

`from_config` **accepts its argument and ignores it.** Silero has no per-deployment startup
parameters — threshold arrives per session on `StreamVAD.session_start`, and the frame size
is fixed by the model. The method exists for interface consistency.

Consequently the `engine.silero:` section in `vad.yaml` (`threshold`,
`min_speech_duration_ms`, `min_silence_duration_ms`, `optimal_frame_ms`) is **not consumed
at construction**. If you make any of it meaningful, update the YAML comments in the same
change.

## Pitfalls

- Changing `optimal_frame_ms` also shifts `plugin-vad`'s logging debounce (3 frames ≈ 96 ms,
  10 frames ≈ 320 ms), which is expressed in frames.
- `torch` is the heaviest dependency in the workspace — which is exactly why this engine is
  a separate repo from `plugin-vad`.

## Test suite status

**Needs a real `torch` install.** `tests/test_silero_engine.py` does a module-level
`import torch` and patches the Silero calls, so it cannot run under the light `--no-deps`
setup the other engines use. `../docs/development/workspace.md` records it as not-run for
that reason. Either install `torch` or state explicitly that the suite was skipped.

## Do not

- Share one model instance across sessions.
- Feed the model anything other than 512-sample frames.
- Remove the 30-second buffer cap.
