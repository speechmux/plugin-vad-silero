"""Integration tests for concurrent StreamVAD streams.

Verifies that multiple sessions run through VADPluginServicer concurrently
without state bleed, using both the SileroVADEngine (mocked model) and the
DummyVADEngine side-by-side.

gRPC assigns one thread per bidi stream.  These tests replicate that
threading model by launching each stream in its own thread.
"""

from __future__ import annotations

import struct
import threading
from unittest.mock import MagicMock, patch

import torch

from speechmux_plugin_vad.engine.dummy import DummyVADEngine
from speechmux_plugin_vad.service.vad_servicer import VADPluginServicer
from stt_proto.vad.v1 import vad_pb2


# ── helpers ────────────────────────────────────────────────────────────────

def _pcm_bytes(num_samples: int = 512, amplitude: int = 8000) -> bytes:
    return struct.pack(f"<{num_samples}h", *([amplitude] * num_samples))


def _make_context() -> MagicMock:
    ctx = MagicMock()
    ctx.abort.side_effect = RuntimeError("abort called")
    return ctx


def _stream_messages(session_id: str, num_frames: int) -> list[vad_pb2.VADRequest]:
    messages: list[vad_pb2.VADRequest] = [
        vad_pb2.VADRequest(
            session_start=vad_pb2.SessionStart(session_id=session_id, threshold=0.5)
        )
    ]
    for i in range(num_frames):
        messages.append(
            vad_pb2.VADRequest(
                pcm_data=_pcm_bytes(512),
                sample_rate=16000,
                sequence_number=i + 1,
            )
        )
    messages.append(vad_pb2.VADRequest(session_end=vad_pb2.SessionEnd()))
    return messages


def _run_stream(
    svc: VADPluginServicer,
    session_id: str,
    num_frames: int,
    results: dict[str, list],
    errors: list,
) -> None:
    try:
        ctx = _make_context()
        responses = list(svc.StreamVAD(iter(_stream_messages(session_id, num_frames)), ctx))
        results[session_id] = responses
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


def _make_silero_svc() -> VADPluginServicer:
    """Return a VADPluginServicer backed by a mocked SileroVADEngine."""
    def _make_model() -> MagicMock:
        model = MagicMock()
        model.return_value = torch.tensor(0.9)
        model.eval = MagicMock()
        model.reset_states = MagicMock()
        return model

    with patch(
        "speechmux_plugin_vad_silero.silero_engine.SileroVADEngine._load_model",
        side_effect=_make_model,
    ):
        from speechmux_plugin_vad_silero.silero_engine import SileroVADEngine

        engine = SileroVADEngine()

    # _clone_model is called per session (outside the patch scope above).
    # Replace it on the instance so real torch.jit.load is never triggered.
    engine._clone_model = _make_model  # type: ignore[method-assign]
    return VADPluginServicer(engine)


def _join_all(threads: list[threading.Thread], timeout: float = 10.0) -> None:
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)


# ── Silero: concurrent streams ─────────────────────────────────────────────

def test_silero_concurrent_streams_all_complete():
    """N concurrent Silero streams each receive exactly num_frames responses."""
    svc = _make_silero_svc()
    num_sessions, num_frames = 10, 8
    results: dict[str, list] = {}
    errors: list = []

    threads = [
        threading.Thread(
            target=_run_stream,
            args=(svc, f"sil-{i}", num_frames, results, errors),
        )
        for i in range(num_sessions)
    ]
    _join_all(threads)

    assert not errors, f"stream errors: {errors}"
    assert len(results) == num_sessions
    for sid, responses in results.items():
        assert len(responses) == num_frames, f"{sid}: got {len(responses)}, want {num_frames}"


def test_silero_active_count_returns_to_zero():
    """Active session counter is 0 after all concurrent Silero streams finish."""
    svc = _make_silero_svc()
    results: dict[str, list] = {}
    errors: list = []

    threads = [
        threading.Thread(target=_run_stream, args=(svc, f"sil-{i}", 4, results, errors))
        for i in range(8)
    ]
    _join_all(threads)

    assert not errors
    health = svc.HealthCheck(vad_pb2.Empty(), _make_context())
    assert health.active == 0


# ── Dummy: concurrent streams ──────────────────────────────────────────────

def test_dummy_concurrent_streams_all_complete():
    """N concurrent Dummy streams each receive exactly num_frames responses."""
    svc = VADPluginServicer(DummyVADEngine(speech_prob=0.9))
    num_sessions, num_frames = 20, 10
    results: dict[str, list] = {}
    errors: list = []

    threads = [
        threading.Thread(
            target=_run_stream,
            args=(svc, f"dum-{i}", num_frames, results, errors),
        )
        for i in range(num_sessions)
    ]
    _join_all(threads)

    assert not errors
    assert len(results) == num_sessions
    for sid, responses in results.items():
        assert len(responses) == num_frames, f"{sid}: got {len(responses)}, want {num_frames}"


def test_dummy_session_states_independent_under_concurrency():
    """frame_count does not bleed across concurrent Dummy sessions (cycle mode).

    DummyVADEngine is configured with a 200 ms speech window followed by a
    200 ms silence window (10 frames each at 20 ms/frame).  Each session sends
    exactly 10 frames — all within the speech window.  If state bled between
    sessions, some sessions would advance into the silence window and report
    is_speech=False.
    """
    svc = VADPluginServicer(DummyVADEngine(speech_prob=0.9, speech_ms=200, silence_ms=200))
    num_sessions, num_frames = 10, 10
    results: dict[str, list] = {}
    errors: list = []

    threads = [
        threading.Thread(
            target=_run_stream,
            args=(svc, f"cyc-{i}", num_frames, results, errors),
        )
        for i in range(num_sessions)
    ]
    _join_all(threads)

    assert not errors
    for sid, responses in results.items():
        assert len(responses) == num_frames
        speech_count = sum(1 for r in responses if r.is_speech)
        assert speech_count == num_frames, (
            f"{sid}: expected {num_frames} speech frames, got {speech_count}. "
            "Possible frame_count state bleed."
        )


# ── mixed: Silero + Dummy concurrent ──────────────────────────────────────

def test_silero_and_dummy_streams_run_concurrently():
    """Silero and Dummy servicers handle interleaved streams without errors."""
    silero_svc = _make_silero_svc()
    dummy_svc = VADPluginServicer(DummyVADEngine(speech_prob=0.85))

    num_sessions, num_frames = 5, 6
    silero_results: dict[str, list] = {}
    dummy_results: dict[str, list] = {}
    errors: list = []

    threads = [
        threading.Thread(
            target=_run_stream,
            args=(silero_svc, f"sil-{i}", num_frames, silero_results, errors),
        )
        for i in range(num_sessions)
    ] + [
        threading.Thread(
            target=_run_stream,
            args=(dummy_svc, f"dum-{i}", num_frames, dummy_results, errors),
        )
        for i in range(num_sessions)
    ]
    _join_all(threads)

    assert not errors
    assert len(silero_results) == num_sessions
    assert len(dummy_results) == num_sessions
    for sid, responses in {**silero_results, **dummy_results}.items():
        assert len(responses) == num_frames, f"{sid}: got {len(responses)}, want {num_frames}"
