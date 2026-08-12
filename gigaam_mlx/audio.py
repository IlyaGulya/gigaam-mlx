"""Audio loading and mel spectrogram computation (no PyTorch dependency)."""

import os
import selectors
import shutil
import subprocess
import time

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 320
HOP_LENGTH = 160
WIN_LENGTH = 320


def _stop_process(process: subprocess.Popen) -> None:
    """Stop ffmpeg and reap it after an interrupted or rejected read."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def load_audio(
    path: str,
    sr: int = SAMPLE_RATE,
    *,
    timeout_s: float | None = None,
    max_duration_s: float | None = None,
    max_input_bytes: int | None = None,
) -> np.ndarray:
    """
    Load audio from any file (video, audio) via ffmpeg.

    Returns mono float32 PCM at ``sr`` normalized to [-1, 1]. Optional limits
    are useful when the path comes from an untrusted daemon request.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install it: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)"
        )

    if sr <= 0:
        raise ValueError("sample rate must be positive")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if max_duration_s is not None and max_duration_s <= 0:
        raise ValueError("maximum duration must be positive")
    if max_input_bytes is not None:
        if max_input_bytes <= 0:
            raise ValueError("maximum input size must be positive")
        if os.path.getsize(path) > max_input_bytes:
            raise ValueError("audio input size exceeds max_input_bytes")

    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr), "-",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        _stop_process(process)
        raise RuntimeError("ffmpeg pipes were not created")

    output = bytearray()
    error_output = bytearray()
    max_output_bytes = (
        int(max_duration_s * sr) * 2 if max_duration_s is not None else None
    )
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, data="stdout")
    selector.register(process.stderr, selectors.EVENT_READ, data="stderr")

    try:
        while selector.get_map():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"ffmpeg timed out after {timeout_s} seconds")
            timeout = None if remaining is None else remaining
            events = selector.select(timeout)
            if not events:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"ffmpeg timed out after {timeout_s} seconds")
                continue

            for key, _ in events:
                data = os.read(key.fileobj.fileno(), 64 * 1024)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    if len(error_output) < 200:
                        error_output.extend(data[:200 - len(error_output)])
                else:
                    output.extend(data)
                    if (
                        max_output_bytes is not None
                        and len(output) > max_output_bytes
                    ):
                        raise ValueError(
                            "decoded audio exceeds the configured maximum duration"
                        )

        returncode = process.wait()
        if returncode != 0:
            error = error_output.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed to load audio: {error}")
    except BaseException:
        _stop_process(process)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()

    return np.frombuffer(output, dtype=np.int16).astype(np.float32) / 32768.0


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
    if audio.ndim != 1:
        raise ValueError("audio must be mono PCM with shape (samples,)")
    if len(audio) < WIN_LENGTH:
        audio = np.pad(audio, (0, WIN_LENGTH - len(audio)))

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
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    chunk_samples = int(max_chunk_sec * sr)
    if chunk_samples <= 0:
        raise ValueError("maximum chunk duration must be positive")
    min_silence = max(1, int(0.3 * sr))
    silence_kernel = np.ones(min_silence) / min_silence
    total = len(audio)
    chunks = []
    start = 0

    while start < total:
        end = min(start + chunk_samples, total)
        if end < total:
            search_start = max(start + chunk_samples // 2, start)
            window = np.abs(audio[search_start:end])
            if len(window) > min_silence:
                # A prefix sum computes the same sliding-window energy in O(n)
                # instead of np.convolve's O(n * min_silence) work.
                prefix = np.empty(len(window) + 1, dtype=np.float64)
                prefix[0] = 0.0
                np.cumsum(window, dtype=np.float64, out=prefix[1:])
                energy = prefix[min_silence:] - prefix[:-min_silence]
                approximate_best = int(np.argmin(energy))

                # Refine around the minimum using the old calculation so tiny
                # summation-order differences do not move nearby boundaries.
                # Seventeen candidates keep the algorithm linear in length.
                radius = 8
                first = max(0, approximate_best - radius)
                last = min(len(energy), approximate_best + radius + 1)
                local_window = window[first:last + min_silence - 1]
                local_energy = np.convolve(
                    local_window, silence_kernel, mode="valid"
                )
                best = first + int(np.argmin(local_energy))
                end = search_start + best + min_silence // 2

        chunks.append({
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / sr,
            "end_sec": end / sr,
        })
        start = end

    return chunks
