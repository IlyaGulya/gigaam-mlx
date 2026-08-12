import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from gigaam_mlx import FRAME_DURATION_S
from gigaam_mlx.model import GigaAMMLX


class IdentityProjection:
    def __call__(self, value):
        return value


class DecodeFramesTest(unittest.TestCase):
    def test_frame_duration_is_public(self):
        self.assertEqual(FRAME_DURATION_S, 0.04)

    def test_ctc_frames_follow_collapse_and_blank_filtering(self):
        labels = [3, 1, 1, 3, 1, 2, 2, 3, 0]
        logits = np.full((1, len(labels), 4), -1.0, dtype=np.float32)
        logits[0, np.arange(len(labels)), labels] = 1.0
        normalize_values = []

        def head(encoded, normalize=True):
            normalize_values.append(normalize)
            return mx.array(logits)

        model = SimpleNamespace(
            head=head,
            num_classes=4,
        )
        encoded = mx.zeros((1, 1, len(labels)))

        tokens = GigaAMMLX._ctc_decode(model, encoded, len(labels))
        framed = []
        framed_tokens = GigaAMMLX._ctc_decode(
            model, encoded, len(labels), token_frames=framed
        )

        self.assertEqual(tokens, [1, 1, 2, 0])
        self.assertEqual(framed_tokens, tokens)
        self.assertEqual(framed, [(1, 1), (1, 4), (2, 5), (0, 8)])
        self.assertEqual(tokens, [token for token, _ in framed])
        self.assertEqual(normalize_values, [False, False])

    def test_public_decode_methods_have_unambiguous_results(self):
        calls = []

        def ctc_decode(encoded, seq_len, token_frames=None):
            calls.append((encoded, seq_len, token_frames is not None))
            if token_frames is not None:
                token_frames.append((7, 3))
            return [7]

        model = SimpleNamespace(model_type="ctc", _ctc_decode=ctc_decode)
        encoded = object()

        self.assertEqual(GigaAMMLX.decode(model, encoded, 4), [7])
        self.assertEqual(
            GigaAMMLX.decode_with_frames(model, encoded, 4), [(7, 3)]
        )
        self.assertEqual(calls, [(encoded, 4, False), (encoded, 4, True)])

    def test_rnnt_uses_scan_offset_and_reuses_frame_for_extra_symbols(self):
        # Blank on frames 0 and 1, then token 0 and token 1 on frame 2.
        weights = {
            "Wx": np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            "Wh": np.zeros((1, 4), dtype=np.float32),
            "b": np.array([10.0, -10.0, 0.0, 10.0], dtype=np.float32),
            "embed": np.array([[4.0], [-4.0], [0.0]], dtype=np.float32),
            "Wp": np.array([[3.0]], dtype=np.float32),
            "bp": np.zeros(1, dtype=np.float32),
            "Wo": np.array([[1.0, 2.0, 0.0]], dtype=np.float32),
            "bo": np.array([0.0, -3.0, 1.0], dtype=np.float32),
        }
        model = SimpleNamespace(
            decoder=SimpleNamespace(blank_id=2, pred_hidden=1),
            joint=SimpleNamespace(enc_proj=IdentityProjection()),
            _decode_weights=lambda: weights,
        )
        encoded = mx.array([[[0.0, 0.0, 2.0, 0.0, 0.0]]])

        tokens = GigaAMMLX._rnnt_decode(model, encoded, 5, lookahead=3)
        framed = []
        framed_tokens = GigaAMMLX._rnnt_decode(
            model, encoded, 5, lookahead=3, token_frames=framed
        )

        self.assertEqual(tokens, [0, 1])
        self.assertEqual(framed_tokens, tokens)
        self.assertEqual(framed, [(0, 2), (1, 2)])
        self.assertEqual(tokens, [token for token, _ in framed])


if __name__ == "__main__":
    unittest.main()
