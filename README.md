# speechmux/plugin-vad-silero

Silero VAD engine plugin for SpeechMux. Depends on `speechmux-plugin-vad` for the `VADEngine` Protocol. `SileroVADEngine` inherits from `VADEngine` and is registered via `entry_points` so the base server discovers it automatically when `server.engine: silero` is set in the config YAML.

## Features

- **Silero VAD** model — prefers ONNX backend (`onnxruntime`), falls back to TorchScript
- **32 ms frame boundary** (512 samples @ 16 kHz) — matches Silero's internal window
- **Per-session hidden state isolation** — TorchScript: `deepcopy`; ONNX: model reload (no state bleed between concurrent sessions)
- **Auto-registered** via `entry_points("speechmux.vad_engine")["silero"]`

## Requirements

- Python 3.10+
- `speechmux-plugin-vad` base package
- `numpy >= 1.24`
- PyTorch (`torch >= 2.1.0`) + `torchaudio >= 2.1.0`
- `silero-vad >= 5.0` (model weights auto-downloaded on first run)
- `onnxruntime` (optional, recommended for better performance)

## Install

```bash
# Base package first
pip install speechmux-plugin-vad

# Silero engine
pip install -e ".[dev]"
```

## Run

Run from the workspace root (the config file lives in `plugin-vad/config/`):

```bash
python -m speechmux_plugin_vad.main --config plugin-vad/config/vad.yaml
```

Or from the `plugin-vad/` directory:

```bash
cd ../plugin-vad
python -m speechmux_plugin_vad.main --config config/vad.yaml
```

`server.engine: silero` in `vad.yaml` triggers automatic discovery of this package via `entry_points`.

## Test

```bash
python -m pytest tests/ -v
```

### Test Coverage

- **Unit tests** (`test_silero_engine.py`): frame processing, probability thresholds, session state isolation, RMS computation, model metadata, buffered sample accumulation
- **Integration tests** (`test_concurrent_streams.py`): 10+ concurrent `StreamVAD` streams through the gRPC servicer, verifying no state bleed between sessions in a multi-threaded environment; mixed Silero + Dummy engine concurrency; active session counter accuracy

## Configuration (vad.yaml)

```yaml
engine:
  silero:
    threshold: 0.5               # Speech probability threshold (0.0–1.0). Lower = more sensitive.
    min_speech_duration_ms: 250   # Min speech segment (ms). Shorter bursts are ignored.
    min_silence_duration_ms: 100  # Min silence (ms) before closing a speech segment.
    optimal_frame_ms: 32          # Frame size fed to Silero (ms). 32ms = 512 samples @ 16kHz.
```

> **Note**: `threshold` is passed per-session via the `StreamVAD` RPC request, not read from
> this config section. The fields above are accepted by `from_config()` for forward-compatibility
> but are currently unused by the engine.

## How It Works

1. On session start, `create_session_state(threshold)` clones the Silero model per session (TorchScript: `deepcopy`; ONNX: reload from disk) to create an isolated instance with its own hidden state
2. Each `process_frame(state, pcm_data, sample_rate)` feeds audio through the session's private model instance
3. Returns `(is_speech, speech_probability, chunk_rms)` — Core's EPD Controller uses `is_speech` to detect utterance boundaries
4. Sub-frame audio (< 512 samples) is buffered internally until a full frame is accumulated

## entry_points Registration

```toml
[project.entry-points."speechmux.vad_engine"]
silero = "speechmux_plugin_vad_silero.silero_engine:SileroVADEngine"
```

After installation, set `server.engine: silero` in `vad.yaml` — the registry discovers it automatically without any code changes to the base plugin.

## License

MIT
