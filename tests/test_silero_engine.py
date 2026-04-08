"""Tests for SileroVADEngine.

These tests run without a real Silero model by patching the torch/silero
imports, so the test suite passes in CI without GPU or model weights.
"""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import torch
import pytest


def _pcm_bytes(n_samples: int = 512, amplitude: int = 8000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


def _silence_bytes(n_samples: int = 512) -> bytes:
    return bytes(n_samples * 2)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_silero_model():
    """A mock Silero model that always returns 0.9 probability."""
    model = MagicMock()
    model.return_value = torch.tensor(0.9)
    model.eval = MagicMock()
    model.reset_states = MagicMock()
    return model


def _make_mock_model() -> MagicMock:
    """Return a fresh mock Silero model that always returns 0.9 probability."""
    model = MagicMock()
    model.return_value = torch.tensor(0.9)
    model.eval = MagicMock()
    model.reset_states = MagicMock()
    return model


@pytest.fixture()
def engine():
    # Return a new model instance on every _load_model call so that isolation
    # tests remain valid regardless of whether we use deepcopy (TorchScript) or
    # reload (ONNX).
    with (
        patch("speechmux_plugin_vad_silero.silero_engine.SileroVADEngine._load_model",
              side_effect=_make_mock_model),
    ):
        from speechmux_plugin_vad_silero.silero_engine import SileroVADEngine
        yield SileroVADEngine()


# ── create_session_state ─────────────────────────────────────────────────

def test_create_session_state_returns_isolated_copy(engine):
    """Each session must get its own model instance (isolated from other sessions)."""
    state1 = engine.create_session_state(0.5)
    state2 = engine.create_session_state(0.5)
    assert state1.model is not state2.model


def test_create_session_state_threshold(engine):
    state = engine.create_session_state(0.7)
    assert state.threshold == pytest.approx(0.7)


# ── process_frame ────────────────────────────────────────────────────────

def test_process_frame_returns_tuple(engine):
    state = engine.create_session_state(0.5)
    result = engine.process_frame(state, _pcm_bytes(512), 16000)
    assert isinstance(result, tuple) and len(result) == 3


def test_process_frame_is_speech_true_when_prob_above_threshold(engine):
    state = engine.create_session_state(threshold=0.5)
    is_speech, prob, _ = engine.process_frame(state, _pcm_bytes(512), 16000)
    assert prob == pytest.approx(0.9)
    assert is_speech is True


def test_process_frame_is_speech_false_when_threshold_high(engine):
    state = engine.create_session_state(threshold=0.95)
    is_speech, prob, _ = engine.process_frame(state, _pcm_bytes(512), 16000)
    assert is_speech is False


def test_silence_gives_zero_rms(engine):
    state = engine.create_session_state(0.5)
    _, _, chunk_rms = engine.process_frame(state, _silence_bytes(512), 16000)
    assert chunk_rms == pytest.approx(0.0)


# ── session isolation ──────────────────────────────────────────────────────

def test_session_states_are_independent(engine):
    """Frame buffers must not bleed across sessions."""
    state1 = engine.create_session_state(0.5)
    state2 = engine.create_session_state(0.5)

    engine.process_frame(state1, _pcm_bytes(256), 16000)  # partial frame in state1
    # state2 buffer must still be empty
    assert state2.buffered_samples == 0


def test_buffered_samples_accumulate_until_frame_size(engine):
    state = engine.create_session_state(0.5)
    # 256 samples — not yet a full 512-sample frame
    engine.process_frame(state, _pcm_bytes(256), 16000)
    # Silero processes nothing yet — buffered_samples should be 0 or less than 512
    # (may be 0 if the frame was consumed, or 256 if not yet)
    assert state.buffered_samples < 512


# ── engine metadata ────────────────────────────────────────────────────────

def test_model_name(engine):
    assert engine.model_name == "silero_vad"


def test_optimal_frame_ms(engine):
    assert engine.optimal_frame_ms == 32


def test_supported_sample_rates(engine):
    assert 16000 in engine.supported_sample_rates
