"""CLI and API for transcribing audio/video files with GigaAM MLX."""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import mlx.core as mx
import numpy as np

from .audio import SAMPLE_RATE, compute_mel, load_audio, split_audio
from .model import DEFAULT_CACHE_LIMIT, GigaAMMLX


@dataclass(frozen=True)
class DecodedChunk:
    """Token emissions and source boundaries for one decoded audio chunk."""

    start: float
    end: float
    emissions: list[tuple[int, int]]


def _validate_audio(audio: np.ndarray) -> None:
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a NumPy array")
    if audio.ndim != 1:
        raise ValueError("audio must be mono PCM with shape (samples,)")


def _decode_chunks(
    audio: np.ndarray,
    model: GigaAMMLX,
    chunks: list[dict],
) -> Iterator[DecodedChunk]:
    for chunk in chunks:
        chunk_audio = audio[chunk["start_sample"]:chunk["end_sample"]]
        mel = compute_mel(chunk_audio)
        encoded, seq_len = model.encode(mx.array(mel[np.newaxis]))
        mx.eval(encoded)
        emissions = model.decode_with_frames(encoded, seq_len)
        yield DecodedChunk(
            start=chunk["start_sec"],
            end=chunk["end_sec"],
            emissions=emissions,
        )


def iter_decode_audio(
    audio: np.ndarray,
    model: GigaAMMLX,
    *,
    max_chunk_sec: float = 20.0,
) -> Iterator[DecodedChunk]:
    """Decode 16kHz mono PCM lazily, yielding one chunk at a time."""
    _validate_audio(audio)
    chunks = split_audio(audio, max_chunk_sec=max_chunk_sec, sr=SAMPLE_RATE)
    return _decode_chunks(audio, model, chunks)


def transcribe_audio(
    audio: np.ndarray,
    model: GigaAMMLX,
    tokenizer,
    *,
    verbose: bool = True,
    max_chunk_sec: float = 20.0,
) -> list[dict]:
    """Transcribe mono float PCM at ``SAMPLE_RATE`` supplied by the caller."""
    _validate_audio(audio)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    started = time.perf_counter()
    log(f"Audio: {len(audio) / SAMPLE_RATE:.1f}s")
    chunks = split_audio(audio, max_chunk_sec=max_chunk_sec, sr=SAMPLE_RATE)
    log(f"Split into {len(chunks)} chunks")

    segments = []
    for index, decoded in enumerate(_decode_chunks(audio, model, chunks)):
        token_ids = [token_id for token_id, _ in decoded.emissions]
        text = tokenizer.decode(token_ids)

        if text.strip():
            segment = {
                "start": decoded.start,
                "end": decoded.end,
                "text": text,
            }
            segments.append(segment)
            log(
                f"  [{format_srt_time(segment['start'])} -> "
                f"{format_srt_time(segment['end'])}] {text}"
            )

        if verbose and (index + 1) % 10 == 0:
            log(f"  ... {index + 1}/{len(chunks)} chunks")

    elapsed = time.perf_counter() - started
    log(f"Transcribed in {elapsed:.1f}s ({len(segments)} segments)")
    return segments


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(seg['start'])} --> "
                f"{format_srt_time(seg['end'])}\n"
            )
            f.write(f"{seg['text'].strip()}\n\n")


def transcribe_file(
    audio_path: str,
    model: Optional[GigaAMMLX] = None,
    tokenizer=None,
    model_type: str = "ctc",
    repo_id: Optional[str] = None,
    verbose: bool = True,
    cache_limit: Optional[int] = DEFAULT_CACHE_LIMIT,
    load_timeout_s: Optional[float] = None,
    max_duration_s: Optional[float] = None,
    max_input_bytes: Optional[int] = None,
) -> list[dict]:
    """
    Transcribe an audio or video file.

    Args:
        audio_path: Path to audio/video file
        model: Pre-loaded model (loads from HF if None)
        tokenizer: Pre-loaded tokenizer
        model_type: "ctc" (fast) or "rnnt" (higher quality)
        repo_id: HuggingFace repo ID (auto-selected if None)
        verbose: Print progress
        cache_limit: Cap MLX's buffer cache, in bytes. Pass None to leave
            MLX's default (unbounded) behaviour untouched.
        load_timeout_s: Optional ffmpeg timeout in seconds.
        max_duration_s: Reject decoded audio longer than this many seconds.
        max_input_bytes: Reject input files larger than this many bytes.

    Returns:
        List of segments with 'start', 'end', 'text' keys
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    if model is None or tokenizer is None:
        from . import load_model
        model, tokenizer = load_model(
            model_type=model_type, repo_id=repo_id, cache_limit=cache_limit
        )
    elif cache_limit is not None:
        # Pre-loaded model, so load_model did not run — apply the cap here.
        mx.set_cache_limit(cache_limit)

    log(f"Loading audio: {os.path.basename(audio_path)}")
    audio = load_audio(
        audio_path,
        timeout_s=load_timeout_s,
        max_duration_s=max_duration_s,
        max_input_bytes=max_input_bytes,
    )
    return transcribe_audio(
        audio, model=model, tokenizer=tokenizer, verbose=verbose
    )


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio/video with GigaAM MLX"
    )
    parser.add_argument("input", help="Path to audio or video file")
    parser.add_argument(
        "--model-type", default="ctc", choices=["ctc", "rnnt"],
        help="Model variant: ctc (fast) or rnnt (higher quality)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--model", default=None, help="HF repo ID or local model path")
    parser.add_argument("--format", choices=["srt", "txt", "both"], default="both")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--cache-limit-gb", type=float, default=DEFAULT_CACHE_LIMIT / 1024**3,
        help="Cap MLX's buffer cache in GB (0 disables the cap)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return

    output_dir = args.output_dir or os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    segments = transcribe_file(
        input_path,
        model_type=args.model_type,
        repo_id=args.model,
        verbose=not args.quiet,
        cache_limit=(
            int(args.cache_limit_gb * 1024**3) if args.cache_limit_gb > 0 else None
        ),
    )

    if not segments:
        print("No speech detected.")
        return

    if args.format in ("srt", "both"):
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        write_srt(segments, srt_path)
        print(f"Saved: {srt_path}")

    if args.format in ("txt", "both"):
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(s["text"].strip() for s in segments))
        print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
