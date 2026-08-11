# gigaam-mlx

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3%2FM4-black?logo=apple)](https://github.com/ml-explore/mlx)
[![HuggingFace CTC](https://img.shields.io/badge/%F0%9F%A4%97-CTC_Model-yellow)](https://huggingface.co/aystream/GigaAM-v3-e2e-ctc-mlx)
[![HuggingFace RNNT](https://img.shields.io/badge/%F0%9F%A4%97-RNNT_Model-yellow)](https://huggingface.co/aystream/GigaAM-v3-e2e-rnnt-mlx)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01192-b31b1b.svg)](https://arxiv.org/abs/2506.01192)

> Fork of [aystream/gigaam-mlx](https://github.com/aystream/gigaam-mlx) — **~4x faster on
> long files, with bounded memory**

Upstream is tuned for single short clips. On a full-length recording split into hundreds of
chunks, two things go wrong: memory grows without bound, and the RNNT decode dominates the
runtime. This fork fixes both. **Every change is verified to leave the transcript byte-identical**
on a 52-minute call (191 segments, 4572 words).

| | upstream | this fork |
|---|---|---|
| 52-min recording (RNNT) | ~56s (55x realtime) | **~14s (~220x realtime)** |
| Peak memory over that run | ~49 GB | **~2 GB** |
| Transcript | — | byte-identical |

```bash
pip install git+https://github.com/IlyaGulya/gigaam-mlx.git
```

The API is unchanged, so this is a drop-in replacement for upstream.

## What this fork changes

### Memory: bounded buffer cache

MLX keeps freed Metal buffers in an unbounded pool. Transcribing chunk after chunk accumulates
them for the whole file — about 0.25 GB per chunk, reaching **27 GB by chunk 100 and 49 GB by the
end of a 52-minute file**. This memory is reclaimable rather than leaked, so it never triggers an
OOM, but it counts toward the process footprint and drives system-wide memory pressure.

It is also invisible to the usual tools: `ps` RSS stays flat at ~1.5 GB because unified-memory
Metal buffers are not counted there. Activity Monitor's "Memory" column does show it.

`load_model` now calls `mx.set_cache_limit(2 GB)`. Peak usage within a single chunk is ~1.7 GB, so
a 2 GB cap preserves buffer reuse inside a chunk while dropping the cross-chunk accumulation.
The limit lives in `load_model` rather than `transcribe_file` so that callers driving
`model.encode` / `model.decode` directly are covered too.

```python
load_model(cache_limit=None)             # opt out, restore upstream behaviour
transcribe_file(path, cache_limit=None)  # same, via the API
load_model(cache_limit=4 * 1024**3)      # or pick your own cap, in bytes
```

```bash
gigaam-mlx recording.mkv --cache-limit-gb 4   # 0 disables the cap
```

### Speed: ~4x faster on long files

Measured on a 52-minute recording (203 chunks), M1 Max, RNNT:

| Stage | Wall | RTF | Decode |
|---|---|---|---|
| upstream | ~56s | 55x | 42.1s |
| + CPU-stream decode, no log_softmax | 30.1s | 103x | 13.5s |
| + encoder projection hoisted | 22.0s | 141x | 8.3s |
| + NumPy decode loop | 17.3s | 178x | 2.7s |
| + mel filterbank cache | 15.4s | 200x | 2.2s |
| + attention and depthwise conv | ~14s | ~220x | 2.2s |

**RNNT decode (42.1s → 2.2s, 19x).** The greedy loop ran on the GPU one token at a time, so every
step paid a GPU sync to read back a single argmax. It now runs entirely in NumPy on the CPU:
blank runs are decoded as batched spans instead of step-by-step, the joint network's encoder
projection is computed once per chunk instead of per step, and `log_softmax` is skipped since
greedy argmax is invariant to it.

**Mel spectrogram.** `librosa.feature.melspectrogram` rebuilds the filterbank and window on every
call, which dominates the cost for short chunks. Both are now cached per sample rate and the STFT
is done directly with a strided view.

**Attention.** Each Conformer layer normalized the same residual three times to feed q, k and v,
and the attention block routed all three through a `(T, B, H, D)` transpose pair just to apply
RoPE. Since q and k are the same tensor in self-attention and v is never rotated, one
normalization and one rotation suffice, applied in place on `(B, T, D)`.

**Depthwise convolution — custom Metal kernel.** MLX routes `nn.Conv1d` with `groups == channels`
through its general grouped-convolution path, which suits this shape badly: the k=5 depthwise does
**154x fewer FLOPs** than a 1x1 pointwise conv over the same tensor yet takes just as long,
running ~5x off the bandwidth bound. A plain 1D stencil written with `mx.fast.metal_kernel` is
2.5–3.4x faster in isolation and bit-identical — the five taps accumulate in the same order. It
falls back to `nn.Conv1d` when custom Metal kernels are unavailable or the input is not float32.

### Notes on measurement

Wall-clock numbers on Apple Silicon vary a lot with thermal state — the same run measured
anywhere from 14s to 23s on the same machine within one session. Comparisons above are best-of-N
A/B pairs taken within a single process, which is the only way to get a clean signal. Treat the
absolute figures as indicative and the ratios as meaningful.

### Approaches that were tried and rejected

All measured, none kept. Recorded here so they don't get re-attempted:

| Idea | Result |
|---|---|
| fp16 / bf16 | 0.87x / 0.85x, plus token drift |
| `mx.fast.scaled_dot_product_attention` | 0.31x |
| `mx.compile` | 1.05x steady-state, but 203 distinct shapes force a 1.5s recompile each |
| Padding mel length to a grid (64/256) | 0.71x / 0.95x, and breaks output without attention masking |
| `mx.async_eval` | 0.67x |
| Dropping `mx.eval` between encode and decode | 14% slower |
| Overlapping CPU phases with GPU via threads | 1.00x — the GPU is the critical path throughout |
| Batching chunks along the batch axis | 1.07–1.11x, costs streaming and bit-identity |
| Longer chunks (30s) | 0.73x and loses 1.6% of words |
| Shorter chunks (8–15s) | within noise; per-chunk overhead cancels the quadratic attention saving |
| Pre-scaling q instead of the scores | 1.58x in isolation, 1.00x in the real graph |
| Caching the RoPE tables | no measurable effect |
| Quantization | not attempted — deliberately out of scope for this fork |

The encoder is ~75% of the remaining runtime and its GEMMs already run at ~60% of the machine's
fp32 peak, which is near the practical ceiling for these shapes.

---

*Everything below is upstream's documentation.*

## About

MLX port of [GigaAM-v3](https://github.com/salute-developers/GigaAM) (220M params, Conformer + CTC/RNNT) by Salute Developers. Produces **punctuated, normalized text** directly. No PyTorch required.

<p align="center">
  <img src="assets/benchmark.svg" alt="Benchmark comparison" width="600">
</p>

## Quick Start

```python
from gigaam_mlx import load_model, transcribe

model, tokenizer = load_model()  # auto-downloads from HuggingFace
text = transcribe(model, tokenizer, "meeting.wav")
print(text)
```

## CLI

```bash
# Transcribe any audio/video file (CTC — fast, default)
gigaam-mlx recording.mkv

# Use RNNT for higher quality
gigaam-mlx recording.mkv --model-type rnnt

# Output subtitles
gigaam-mlx call.wav --output-dir ./transcripts --format srt
```

Outputs `.srt` (subtitles) and `.txt` (plain text). Model weights download automatically on first run.

## Performance

MacBook Pro M2 Max, 20-second audio chunk (avg of 3 runs, warmed up):

| Backend | Model | Time | Realtime factor |
|---|---|---|---|
| **MLX (this)** | **v3_e2e_ctc** | **0.06s** | **~330x** |
| **MLX (this)** | **v3_e2e_rnnt** | **0.26s** | **~77x** |
| PyTorch MPS | v3_e2e_rnnt | 0.76s | ~26x |
| PyTorch CPU | v3_e2e_rnnt | 1.13s | ~18x |
| ONNX CPU | v3_e2e_ctc | 1.66s | ~12x |

Full 18-minute video: CTC **21.5s** (~50x realtime), RNNT **25.0s** (~42x realtime).

These are single-chunk figures; for long files see the fork section above.

## Model variants

| Variant | Speed | Quality | Use case |
|---|---|---|---|
| **CTC** (default) | ~330x realtime | Good | Batch processing, speed-critical |
| **RNNT** | ~77x realtime | Better | When accuracy matters most |

```python
# Higher quality with RNNT
model, tokenizer = load_model("rnnt")
```

## Features

- **up to 330x realtime** on Apple Silicon (M1/M2/M3/M4)
- **Russian + English** — recognizes English words/terms in Russian speech
- **Punctuation** built-in — end-to-end model, no post-processing
- **No PyTorch** — pure MLX + librosa + numpy
- **Any format** — video and audio via ffmpeg (mkv, mp4, wav, mp3, ...)
- **Auto-download** — model weights from HuggingFace Hub

## Requirements

- macOS with Apple Silicon (M1+)
- Python >= 3.10
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)

## How it works

<p align="center">
  <img src="https://raw.githubusercontent.com/salute-developers/GigaAM/main/assets/gigaam_scheme.svg" alt="GigaAM architecture" width="700">
  <br>
  <em>GigaAM model family (<a href="https://github.com/salute-developers/GigaAM">source</a>)</em>
</p>

```
Audio/Video → ffmpeg (16kHz mono) → Mel spectrogram (librosa)
    → Conformer encoder (16 layers, 768d, 16 heads, RoPE)
    → CTC/RNNT head → greedy decode → punctuated text
```

The model is a 220M parameter Conformer pretrained on 700,000 hours of Russian speech. The `v3_e2e_ctc` variant produces punctuated, normalized text directly — no language model or post-processing needed.

## Converting weights yourself

```bash
pip install gigaam-mlx[convert]
python -m gigaam_mlx.convert --model v3_e2e_ctc --output-dir ./weights_ctc
python -m gigaam_mlx.convert --model v3_e2e_rnnt --output-dir ./weights_rnnt
```

## Acknowledgments

- [GigaAM](https://github.com/salute-developers/GigaAM) by Salute Developers / SberDevices — original model ([paper](https://arxiv.org/abs/2506.01192), InterSpeech 2025)
- [MLX](https://github.com/ml-explore/mlx) by Apple — ML framework for Apple Silicon
- [ai-sage/GigaAM-v3](https://huggingface.co/ai-sage/GigaAM-v3) — HuggingFace transformers integration

## License

MIT — same as the original GigaAM model.
