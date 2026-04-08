"""Silero VAD engine for speechmux-plugin-vad."""

from __future__ import annotations

import collections
import copy
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from speechmux_plugin_vad.engine.base import VADEngine

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_SILERO_SAMPLE_RATE = 16000
_FRAME_SIZE = 512  # 32 ms at 16 kHz (Silero requirement)
# Safety cap: discard frame_buffer if it exceeds 30 s of audio. Normal Core
# sends 32 ms / 512-sample frames so this cap should never be reached in
# practice.  It guards against misbehaving clients that send very large chunks.
_MAX_BUFFERED_SAMPLES = _SILERO_SAMPLE_RATE * 30

# Silero model type — can be torch.jit.ScriptModule or an ONNX wrapper;
# both are callable with (tensor, sample_rate) signature.
_SileroModel = Callable[..., "torch.Tensor | float | None"]


@dataclass
class _SileroSessionState:
    """Mutable per-session state for the Silero VAD model.

    Attributes:
        model: The session-private Silero model instance.
        threshold: Speech probability threshold (0.0–1.0).
        frame_buffer: Deque of float32 chunks awaiting framing.
        buffered_samples: Total number of samples currently in ``frame_buffer``.
        chunk_offset: Read offset within ``frame_buffer[0]`` (the current head chunk).
    """

    model: _SileroModel
    threshold: float
    # Incoming audio chunk accumulator for 512-sample framing.
    frame_buffer: collections.deque[NDArray[np.float32]] = field(
        default_factory=collections.deque,
    )
    buffered_samples: int = 0
    chunk_offset: int = 0


class SileroVADEngine(VADEngine):
    """Silero VAD engine.

    Loads one base model at startup, then ``deepcopy``s it for each session so
    that hidden RNN states are fully isolated.  Falls back to reloading the
    model if deepcopy fails (some TorchScript builds are non-picklable).

    Audio is accumulated in 512-sample (32 ms @ 16 kHz) windows before being
    passed to the model, matching Silero's required frame size.
    """

    model_name: str = "silero_vad"
    optimal_frame_ms: int = 32
    supported_sample_rates: list[int] = None  # type: ignore[assignment]
    max_concurrent_sessions: int = 50

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SileroVADEngine:
        """Construct from a ``vad.yaml`` ``engine.silero`` section.

        Silero VAD has no startup parameters that vary per-deployment — threshold
        and frame size are passed per-session via the ``StreamVAD`` RPC request.
        This method exists for interface consistency; *config* is accepted but
        currently unused.

        Args:
            config: Dict from the ``engine.silero`` YAML section.  Accepted
                for forward-compatibility; no keys are consumed at engine
                construction time.

        Returns:
            A fully initialised ``SileroVADEngine``.
        """
        return cls()

    def __init__(self) -> None:
        self.supported_sample_rates = [8000, 16000]
        self._base_model = self._load_model()

    # ── public interface ───────────────────────────────────────────────────

    def create_session_state(self, threshold: float) -> _SileroSessionState:
        """Create an isolated model copy for a new session.

        Args:
            threshold: Speech probability threshold (0.0–1.0).

        Returns:
            A ``_SileroSessionState`` with a private model copy and empty frame buffer.
        """
        model = self._clone_model()
        return _SileroSessionState(model=model, threshold=threshold)

    def process_frame(
        self,
        session_state: object,
        pcm_data: bytes,
        sample_rate: int,
    ) -> tuple[bool, float, float]:
        """Run Silero VAD on one audio chunk.

        Accumulates samples until a full 512-sample frame is available,
        then scores the frame.  Returns the max probability across all
        complete frames in this chunk.

        Args:
            session_state: Per-session state from ``create_session_state``.
            pcm_data: Raw PCM S16LE audio bytes.
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            A ``(is_speech, speech_probability, chunk_rms)`` tuple.
        """
        assert isinstance(session_state, _SileroSessionState)
        state = session_state
        audio_float32 = _pcm16_to_float32(pcm_data)

        if sample_rate and sample_rate != _SILERO_SAMPLE_RATE:
            audio_float32 = _resample(audio_float32, sample_rate, _SILERO_SAMPLE_RATE)

        chunk_rms = (
            float(np.sqrt(np.mean(audio_float32 ** 2))) if audio_float32.size > 0 else 0.0
        )

        state.frame_buffer.append(audio_float32)
        state.buffered_samples += audio_float32.size

        if state.buffered_samples > _MAX_BUFFERED_SAMPLES:
            logger.warning(
                "VAD frame_buffer exceeded 30 s cap (%d samples); discarding buffer",
                state.buffered_samples,
            )
            state.frame_buffer.clear()
            state.buffered_samples = 0
            state.chunk_offset = 0
            return False, 0.0, chunk_rms

        max_probability = 0.0
        while state.buffered_samples >= _FRAME_SIZE:
            frame = self._extract_frame(state)
            probability = self._silero_probability(state.model, frame)
            if probability > max_probability:
                max_probability = probability

        is_speech = max_probability >= state.threshold
        return is_speech, max_probability, chunk_rms

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_model() -> _SileroModel:
        """Load Silero VAD model, preferring ONNX backend.

        Returns:
            A callable Silero model (TorchScript or ONNX wrapper).
        """
        from silero_vad import load_silero_vad  # type: ignore[import-untyped]
        try:
            model = load_silero_vad(onnx=True)
        except (ImportError, RuntimeError, OSError):
            logger.warning(
                "ONNX Silero VAD load failed; falling back to TorchScript. "
                "Install onnxruntime for better performance.",
                exc_info=True,
            )
            model = load_silero_vad()
        if hasattr(model, "eval"):
            model.eval()
        if hasattr(model, "reset_states"):
            model.reset_states()
        return model

    def _clone_model(self) -> _SileroModel:
        """Clone the base model for a new session.

        TorchScript modules support deep-copy reliably.  ONNX InferenceSession
        wrappers may share the underlying C++ InferenceSession state after
        deepcopy (shallow copy of the native object), so ONNX models are always
        reloaded from disk to guarantee per-session isolation.

        Returns:
            A cloned or freshly loaded Silero model with reset hidden states.
        """
        try:
            import torch
            is_torchscript = isinstance(self._base_model, torch.jit.ScriptModule)
        except ImportError:
            is_torchscript = False

        if is_torchscript:
            try:
                cloned = copy.deepcopy(self._base_model)
            except Exception:
                logger.warning(
                    "deepcopy of Silero TorchScript model failed; reloading instead",
                    exc_info=True,
                )
                cloned = self._load_model()
        else:
            # ONNX: reload to guarantee isolated InferenceSession state.
            cloned = self._load_model()

        if hasattr(cloned, "eval"):
            cloned.eval()
        if hasattr(cloned, "reset_states"):
            cloned.reset_states()
        return cloned

    @staticmethod
    def _extract_frame(state: _SileroSessionState) -> NDArray[np.float32]:
        """Extract exactly ``_FRAME_SIZE`` samples from the buffer.

        Args:
            state: Session state containing the frame buffer.

        Returns:
            Float32 numpy array of exactly ``_FRAME_SIZE`` samples.
        """
        frame = np.zeros(_FRAME_SIZE, dtype=np.float32)
        samples_filled = 0
        while samples_filled < _FRAME_SIZE and state.frame_buffer:
            chunk = state.frame_buffer[0]
            remaining_samples = chunk.size - state.chunk_offset
            samples_to_take = min(_FRAME_SIZE - samples_filled, remaining_samples)
            end = samples_filled + samples_to_take
            frame[samples_filled:end] = chunk[
                state.chunk_offset: state.chunk_offset + samples_to_take
            ]
            samples_filled += samples_to_take
            state.chunk_offset += samples_to_take
            if state.chunk_offset >= chunk.size:
                state.frame_buffer.popleft()
                state.chunk_offset = 0
        state.buffered_samples -= _FRAME_SIZE
        return frame

    @staticmethod
    def _silero_probability(
        model: _SileroModel, frame: NDArray[np.float32]
    ) -> float:
        """Run the Silero model on one 512-sample frame and return probability.

        Args:
            model: Silero VAD model (TorchScript or ONNX wrapper).
            frame: Float32 numpy array of ``_FRAME_SIZE`` samples.

        Returns:
            Speech probability in [0.0, 1.0].
        """
        import torch

        if frame.size == 0:
            return 0.0
        tensor = torch.from_numpy(frame).unsqueeze(0)
        with torch.no_grad():
            model_output = model(tensor, _SILERO_SAMPLE_RATE)
        if isinstance(model_output, torch.Tensor):
            return float(model_output.mean().item())
        if model_output is None:
            return 0.0
        return float(model_output)


# ── audio utilities ────────────────────────────────────────────────────────

def _pcm16_to_float32(pcm_data: bytes) -> NDArray[np.float32]:
    """Convert PCM S16LE bytes to float32 numpy array in [-1, 1].

    Args:
        pcm_data: Raw PCM S16LE byte buffer.

    Returns:
        Numpy float32 array with sample values normalised to [-1, 1].
    """
    num_samples = len(pcm_data) // 2
    if num_samples == 0:
        return np.empty(0, dtype=np.float32)
    raw = struct.unpack(f"<{num_samples}h", pcm_data[: num_samples * 2])
    samples = np.array(raw, dtype=np.float32)
    return samples / 32768.0


def _resample(
    audio: NDArray[np.float32], src_rate: int, dst_rate: int
) -> NDArray[np.float32]:
    """Resample a float32 numpy array from src_rate to dst_rate.

    Args:
        audio: Input float32 audio samples.
        src_rate: Source sample rate in Hz.
        dst_rate: Target sample rate in Hz.

    Returns:
        Resampled float32 numpy array.
    """
    import torch
    import torchaudio  # type: ignore[import-untyped]

    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, src_rate, dst_rate)
    return resampled.squeeze(0).numpy()
