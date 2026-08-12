"""GigaAM v3 e2e — Conformer encoder + CTC/RNNT head on Apple MLX."""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# MLX keeps freed Metal buffers in an unbounded pool by default, so chunked
# transcription accumulates them for the whole file: ~0.25GB per chunk, 27GB
# by chunk 100 of a 52-minute recording. That memory is reclaimable rather
# than leaked, but it counts toward the process footprint and drives system
# memory pressure until something else needs it.
#
# Peak usage within a single chunk is ~1.7GB, so a 2GB cache preserves buffer
# reuse inside a chunk while dropping the cross-chunk accumulation.
DEFAULT_CACHE_LIMIT = 2 * 1024**3

# A 10 ms mel hop followed by the encoder's 4x convolutional subsampling.
FRAME_DURATION_S = 0.04


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# MLX routes nn.Conv1d with groups == channels through its general grouped-conv
# path, which runs about 5x off the bandwidth bound here: the k=5 depthwise does
# 154x fewer FLOPs than a 1x1 pointwise conv over the same tensor yet takes just
# as long. A plain 1D stencil is ~3.4x faster and, since it accumulates the five
# taps in the same order, bit-identical.
_DEPTHWISE_SOURCE = """
  uint gid = thread_position_in_grid.x;
  uint total = shp[0] * shp[1] * shp[2];
  if (gid >= total) return;

  uint C = shp[2];
  uint L = shp[1];
  uint c = gid % C;
  uint t = (gid / C) % L;
  uint b = gid / (C * L);

  float acc = bias[c];
  for (uint k = 0; k < KSZ; ++k) {
    int ti = int(t) + int(k) - int(PADSZ);
    if (ti < 0 || ti >= int(L)) continue;
    acc += x[(b * L + uint(ti)) * C + c] * w[k * C + c];
  }
  out[gid] = acc;
"""

_DEPTHWISE_KERNELS: dict = {}


def _depthwise_kernel(kernel_size: int):
    """Compile (once per kernel size) the depthwise stencil, or None if
    custom Metal kernels are unavailable on this build."""
    if kernel_size not in _DEPTHWISE_KERNELS:
        try:
            _DEPTHWISE_KERNELS[kernel_size] = mx.fast.metal_kernel(
                name=f"depthwise{kernel_size}",
                input_names=["x", "w", "bias", "shp"],
                output_names=["out"],
                source=_DEPTHWISE_SOURCE,
                header=(
                    f"#define KSZ {kernel_size}\n"
                    f"#define PADSZ {(kernel_size - 1) // 2}\n"
                ),
            )
        except Exception:
            _DEPTHWISE_KERNELS[kernel_size] = None
    return _DEPTHWISE_KERNELS[kernel_size]


# ── Rotary Positional Encoding ──────────────────────────────────

def create_rotary_pe(
    length: int, dim: int, base: int = 5000
) -> Tuple[mx.array, mx.array]:
    """Create rotary positional embeddings (cos, sin)."""
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(length, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def _apply_rotary(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    """Apply RoPE to q, k. Input shape: (T, B, H, D)."""
    T = q.shape[0]
    cos = cos[:T, None, None, :]
    sin = sin[:T, None, None, :]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ── Conformer Building Blocks ───────────────────────────────────

class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.kernel_size = kernel_size
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model,
        )
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def _depthwise(self, x: mx.array) -> mx.array:
        kern = _depthwise_kernel(self.kernel_size)
        if kern is None or x.dtype != mx.float32:
            return self.depthwise_conv(x)

        B, L, C = x.shape
        n = B * L * C
        tg = 256
        return kern(
            inputs=[
                x,
                # (C, K, 1) as stored -> (K, C), so each tap is contiguous
                mx.transpose(self.depthwise_conv.weight.squeeze(-1), (1, 0)),
                self.depthwise_conv.bias,
                mx.array([B, L, C], dtype=mx.uint32),
            ],
            grid=(((n + tg - 1) // tg) * tg, 1, 1),
            threadgroup=(tg, 1, 1),
            output_shapes=[(B, L, C)],
            output_dtypes=[x.dtype],
        )[0]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pointwise_conv1(x)
        a, b = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(b)  # GLU
        x = self._depthwise(x)
        x = self.batch_norm(x)
        x = nn.silu(x)
        return self.pointwise_conv2(x)


class RotaryMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int):
        super().__init__()
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)

    def __call__(
        self, query: mx.array, key: mx.array, value: mx.array,
        cos: mx.array, sin: mx.array,
    ) -> mx.array:
        B, T, D = query.shape

        # Apply RoPE to raw input before linear projections. The rotation is a
        # per-position reshape/multiply, so it can be done directly on (B, T, D)
        # without routing through a (T, B, H, D) transpose and back.
        cos_b = mx.broadcast_to(
            cos[:T].reshape(1, T, 1, self.d_k), (1, T, self.h, self.d_k)
        ).reshape(1, T, D)
        sin_b = mx.broadcast_to(
            sin[:T].reshape(1, T, 1, self.d_k), (1, T, self.h, self.d_k)
        ).reshape(1, T, D)

        def rope(x: mx.array) -> mx.array:
            h = x.reshape(B, T, self.h, self.d_k)
            return x * cos_b + _rotate_half(h).reshape(B, T, D) * sin_b

        # Only q and k are rotated; v is passed through untouched. In
        # self-attention q and k are the same tensor, so rotate once.
        if query is key:
            query = key = rope(query)
        else:
            query, key = rope(query), rope(key)

        # Project and compute attention
        q = mx.transpose(self.linear_q(query).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))
        k = mx.transpose(self.linear_k(key).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))
        v = mx.transpose(self.linear_v(value).reshape(B, T, self.h, self.d_k), (0, 2, 1, 3))

        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.d_k)
        out = mx.softmax(scores, axis=-1) @ v

        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, T, self.h * self.d_k)
        return self.linear_out(out)


# ── Conformer Layer & Encoder ───────────────────────────────────

class ConformerLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, conv_kernel_size: int):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RotaryMultiHeadAttention(n_heads, d_model)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        residual = x
        x = self.feed_forward1(self.norm_feed_forward1(x))
        residual = residual + x * self.fc_factor

        normed = self.norm_self_att(residual)
        x = self.self_attn(normed, normed, normed, cos, sin)
        residual = residual + x

        x = self.conv(self.norm_conv(residual))
        residual = residual + x

        x = self.feed_forward2(self.norm_feed_forward2(residual))
        residual = residual + x * self.fc_factor

        return self.norm_out(residual)


class Conv1dSubsampling(nn.Module):
    """2x Conv1d with stride 2 each → 4x subsampling."""

    def __init__(self, feat_in: int, feat_out: int, kernel_size: int = 5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(feat_in, feat_out, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv2 = nn.Conv1d(feat_out, feat_out, kernel_size=kernel_size, stride=2, padding=padding)

    def __call__(self, x: mx.array) -> Tuple[mx.array, int]:
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        return x, x.shape[1]


class ConformerEncoder(nn.Module):
    def __init__(
        self, feat_in: int = 64, n_layers: int = 16, d_model: int = 768,
        n_heads: int = 16, ff_expansion_factor: int = 4,
        conv_kernel_size: int = 5, subs_kernel_size: int = 5,
    ):
        super().__init__()
        self.pre_encode = Conv1dSubsampling(feat_in, d_model, subs_kernel_size)
        self.layers = [
            ConformerLayer(d_model, d_model * ff_expansion_factor, n_heads, conv_kernel_size)
            for _ in range(n_layers)
        ]
        self.rope_dim = d_model // n_heads

    def __call__(self, features: mx.array) -> Tuple[mx.array, int]:
        x, seq_len = self.pre_encode(features)
        cos, sin = create_rotary_pe(seq_len, self.rope_dim)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return mx.transpose(x, (0, 2, 1)), seq_len


# ── CTC Head ────────────────────────────────────────────────────

class CTCHead(nn.Module):
    def __init__(self, feat_in: int = 768, num_classes: int = 257):
        super().__init__()
        self.decoder_layers = nn.Conv1d(feat_in, num_classes, kernel_size=1)

    def __call__(self, encoder_output: mx.array) -> mx.array:
        x = mx.transpose(encoder_output, (0, 2, 1))
        logits = self.decoder_layers(x)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ── RNNT Decoder & Joint ────────────────────────────────────────

class RNNTDecoder(nn.Module):
    def __init__(self, pred_hidden: int = 320, num_classes: int = 1025):
        super().__init__()
        self.pred_hidden = pred_hidden
        self.blank_id = num_classes - 1
        self.embed = nn.Embedding(num_classes, pred_hidden)
        self.lstm = nn.LSTM(pred_hidden, pred_hidden)

    def predict(
        self, x: Optional[mx.array], state: Optional[Tuple[mx.array, mx.array]]
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        if x is not None:
            emb = self.embed(x)
        else:
            emb = mx.zeros((1, 1, self.pred_hidden))
        if state is not None:
            h, c = state
            all_hidden, all_cell = self.lstm(emb, h, c)
        else:
            all_hidden, all_cell = self.lstm(emb)
        return all_hidden, (all_hidden[:, -1, :], all_cell[:, -1, :])


class RNNTJoint(nn.Module):
    def __init__(
        self, enc_hidden: int = 768, pred_hidden: int = 320,
        joint_hidden: int = 320, num_classes: int = 1025,
    ):
        super().__init__()
        self.enc_proj = nn.Linear(enc_hidden, joint_hidden)
        self.pred_proj = nn.Linear(pred_hidden, joint_hidden)
        self.out = nn.Linear(joint_hidden, num_classes)

    def __call__(
        self, enc: mx.array, pred: mx.array, normalize: bool = True
    ) -> mx.array:
        e = mx.expand_dims(self.enc_proj(enc), axis=2)
        p = mx.expand_dims(self.pred_proj(pred), axis=1)
        joint = nn.relu(e + p)
        logits = self.out(joint)
        if not normalize:
            # Greedy decoding only takes an argmax, which log_softmax leaves
            # unchanged — so skip the reduction over all 1025 classes.
            return logits
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


# ── Full Model ──────────────────────────────────────────────────

CTC_CLASSES = 257
RNNT_CLASSES = 1025


class GigaAMMLX(nn.Module):
    """
    GigaAM v3 e2e on Apple MLX.

    220M parameter Conformer encoder for Russian ASR.
    Supports CTC (fast) and RNNT (higher quality) decoding.
    """

    def __init__(self, model_type: str = "ctc"):
        super().__init__()
        self.model_type = model_type
        self.encoder = ConformerEncoder()

        if model_type == "ctc":
            self.num_classes = CTC_CLASSES
            self.head = CTCHead(num_classes=CTC_CLASSES)
        elif model_type == "rnnt":
            self.num_classes = RNNT_CLASSES
            self.decoder = RNNTDecoder(pred_hidden=320, num_classes=RNNT_CLASSES)
            self.joint = RNNTJoint(
                enc_hidden=768, pred_hidden=320,
                joint_hidden=320, num_classes=RNNT_CLASSES,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'ctc' or 'rnnt'.")

    def encode(self, features: mx.array) -> Tuple[mx.array, int]:
        """Run conformer encoder. Input: (B, T, 64) mel spectrogram."""
        return self.encoder(features)

    def decode(self, encoded: mx.array, seq_len: int) -> List[int]:
        """Decode using the model's head (CTC or RNNT)."""
        if self.model_type == "ctc":
            return self._ctc_decode(encoded, seq_len)
        return self._rnnt_decode(encoded, seq_len)

    def decode_with_frames(
        self, encoded: mx.array, seq_len: int,
    ) -> List[Tuple[int, int]]:
        """Decode token IDs paired with zero-based encoder emission frames."""
        token_frames: List[Tuple[int, int]] = []
        if self.model_type == "ctc":
            self._ctc_decode(encoded, seq_len, token_frames=token_frames)
        else:
            self._rnnt_decode(encoded, seq_len, token_frames=token_frames)
        return token_frames

    def _ctc_decode(
        self, encoded: mx.array, seq_len: int,
        token_frames: Optional[List[Tuple[int, int]]] = None,
    ) -> List[int]:
        """CTC greedy decoding — fully vectorized."""
        log_probs = self.head(encoded)
        labels = mx.argmax(log_probs[0, :seq_len, :], axis=-1)
        mx.eval(labels)

        blank_id = self.num_classes - 1
        token_ids: List[int] = []
        prev = blank_id
        for frame, tok in enumerate(labels.tolist()):
            if tok != blank_id and tok != prev:
                token_ids.append(tok)
                if token_frames is not None:
                    token_frames.append((tok, frame))
            prev = tok
        return token_ids

    def _rnnt_decode(
        self, encoded: mx.array, seq_len: int, max_symbols: int = 10,
        lookahead: int = 32,
        token_frames: Optional[List[Tuple[int, int]]] = None,
    ) -> List[int]:
        """
        RNNT greedy decoding.

        Emits the same hypothesis as the naive frame-by-frame loop, but avoids
        one GPU sync per frame. The prediction network output `g` depends only
        on decoder state, which changes only when a non-blank token is emitted
        — and ~77% of frames emit blank. So while the state is fixed, the joint
        for a whole span of upcoming frames is a single batched call, and one
        sync yields every argmax in that span. Decoding resumes frame-by-frame
        from wherever the first non-blank lands.

        A per-step sync costs ~0.72ms against ~0.011ms of actual compute, so
        collapsing the blank runs is worth far more than the redundant joints.

        The loop itself runs in NumPy rather than MLX. Its per-step work is a
        320-unit LSTM cell and a 320->1025 matmul, which is far too small to
        pay for a framework dispatch: the same step costs ~1.08ms on the MLX
        GPU stream, ~0.34ms on the MLX CPU stream, and ~0.06ms in plain NumPy.
        What remained after moving off the GPU was MLX's own scheduling and
        graph-building overhead, so the loop leaves MLX entirely. Only the
        encoder projection stays on the GPU, where it is one batched matmul.

        Skipping log_softmax also drops a reduction over 1025 classes from
        every step — greedy decoding takes an argmax, which is invariant to it.
        """
        blank_id = self.decoder.blank_id
        w = self._decode_weights()

        # The encoder half of the joint does not depend on decoder state, so
        # project every frame once, on the GPU, before dropping into NumPy.
        # Spans restart after each emission, so otherwise the same frames get
        # re-projected many times over.
        enc_p_mx = self.joint.enc_proj(encoded[0].T)  # (T, joint_hidden)
        mx.eval(enc_p_mx)
        enc_p = np.array(enc_p_mx, copy=False)

        H = self.decoder.pred_hidden
        hyp: List[int] = []
        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        emb = np.zeros(H, dtype=np.float32)  # zeros stand in for "no label yet"

        def predict(emb_vec, h, c):
            """One LSTM step; returns (pred_proj(g), new_h, new_c)."""
            z = emb_vec @ w["Wx"] + w["b"] + h @ w["Wh"]
            i = _sigmoid(z[:H])
            f = _sigmoid(z[H:2 * H])
            g = np.tanh(z[2 * H:3 * H])
            o = _sigmoid(z[3 * H:])
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            return h_new @ w["Wp"] + w["bp"], h_new, c_new

        t = 0
        pred_p, h_next, c_next = predict(emb, h, c)
        while t < seq_len:
            # Scan ahead over frames while the decoder state stays fixed.
            span = min(lookahead, seq_len - t)
            block = enc_p[t:t + span] + pred_p
            np.maximum(block, 0, out=block)
            preds = np.argmax(block @ w["Wo"] + w["bo"], axis=-1)

            nz = np.flatnonzero(preds != blank_id)
            if nz.size == 0:
                # Entire span was blank — state unchanged, skip it wholesale.
                t += span
                continue

            # Frames before the first non-blank were blank; emit at that frame.
            offset = int(nz[0])
            t += offset
            k = int(preds[offset])
            hyp.append(k)
            if token_frames is not None:
                token_frames.append((k, t))
            h, c = h_next, c_next
            emb = w["embed"][k]
            pred_p, h_next, c_next = predict(emb, h, c)

            # Same frame may emit several symbols; those need real steps.
            for _ in range(max_symbols - 1):
                logits = np.maximum(enc_p[t] + pred_p, 0) @ w["Wo"] + w["bo"]
                k = int(np.argmax(logits))
                if k == blank_id:
                    break
                hyp.append(k)
                if token_frames is not None:
                    token_frames.append((k, t))
                h, c = h_next, c_next
                emb = w["embed"][k]
                pred_p, h_next, c_next = predict(emb, h, c)
            t += 1
        return hyp

    def _decode_weights(self) -> dict:
        """NumPy copies of the RNNT decoder/joint weights, built once."""
        cached = getattr(self, "_decode_w", None)
        if cached is not None:
            return cached
        lstm = self.decoder.lstm
        w = {
            # MLX stores LSTM gates as (i, f, g, o) along the first axis, and
            # its Linear weights as (out, in) — transpose for x @ W.
            "Wx": np.array(lstm.Wx).T.copy(),
            "Wh": np.array(lstm.Wh).T.copy(),
            "b": np.array(lstm.bias),
            "embed": np.array(self.decoder.embed.weight),
            "Wp": np.array(self.joint.pred_proj.weight).T.copy(),
            "bp": np.array(self.joint.pred_proj.bias),
            "Wo": np.array(self.joint.out.weight).T.copy(),
            "bo": np.array(self.joint.out.bias),
        }
        self._decode_w = w
        return w

    # Keep backward compat
    def ctc_decode(self, encoded: mx.array, seq_len: int) -> List[int]:
        return self._ctc_decode(encoded, seq_len)
