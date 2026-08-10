"""Audio loading and mel spectrogram computation (no PyTorch dependency)."""

import subprocess
import shutil

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio from any file (video, audio) via ffmpeg.

    Returns 16kHz mono float32 numpy array normalized to [-1, 1].
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install it: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed to load audio: {e.stderr[:200]}") from e

    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


_MEL_CACHE: dict = {}


def _mel_basis(sr: int):
    """Mel filterbank and analysis window, built once per sample rate.

    librosa.feature.melspectrogram rebuilds both on every call, which is the
    bulk of the cost when transcribing hundreds of short chunks.
    """
    cached = _MEL_CACHE.get(sr)
    if cached is None:
        fb = librosa.filters.mel(
            sr=sr, n_fft=N_FFT, n_mels=N_MELS, htk=True, norm=None
        ).astype(np.float32)
        window = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)
        cached = (fb, window)
        _MEL_CACHE[sr] = cached
    return cached


def compute_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute log-mel spectrogram matching GigaAM's FeatureExtractor.

    Returns (T, 64) float32 array.
    """
    filters, window = _mel_basis(sr)

    # Equivalent to librosa.feature.melspectrogram(center=False, power=2.0),
    # but without rebuilding the filterbank and window on each call.
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN_LENGTH)
    frames = frames[::HOP_LENGTH]
    spectrum = np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=-1)) ** 2
    mel = filters @ spectrum.T
    return np.log(np.clip(mel, 1e-9, 1e9)).astype(np.float32).T  # (T, n_mels)


def split_audio(
    audio: np.ndarray, max_chunk_sec: float = 20.0, sr: int = SAMPLE_RATE
) -> list[dict]:
    """Split audio at silence points into chunks <= max_chunk_sec."""
    chunk_samples = int(max_chunk_sec * sr)
    min_silence = int(0.3 * sr)
    total = len(audio)
    chunks = []
    start = 0

    while start < total:
        end = min(start + chunk_samples, total)
        if end < total:
            search_start = max(start + chunk_samples // 2, start)
            window = np.abs(audio[search_start:end])
            if len(window) > min_silence:
                energy = np.convolve(
                    window, np.ones(min_silence) / min_silence, mode="valid"
                )
                best = np.argmin(energy)
                end = search_start + best + min_silence // 2

        chunks.append({
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / sr,
            "end_sec": end / sr,
        })
        start = end

    return chunks
