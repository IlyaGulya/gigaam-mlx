import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from gigaam_mlx import (
    SAMPLE_RATE,
    DecodedChunk,
    iter_decode_audio,
    transcribe_audio,
    transcribe_file,
)

CHUNKS = [
    {
        "start_sample": 0,
        "end_sample": 400,
        "start_sec": 0.0,
        "end_sec": 0.025,
    },
    {
        "start_sample": 400,
        "end_sample": 800,
        "start_sec": 0.025,
        "end_sec": 0.05,
    },
]


class FakeModel:
    def __init__(self):
        self.encode_calls = 0

    def encode(self, features):
        self.encode_calls += 1
        return mx.zeros((1, 1, 2)), 2

    def decode_with_frames(self, encoded, seq_len):
        if self.encode_calls == 1:
            return [(10, 0), (11, 1)]
        return [(12, 1)]


class FakeTokenizer:
    def decode(self, token_ids):
        return {10: "first", 12: "second"}.get(token_ids[0], "")


class DecodeAudioTest(unittest.TestCase):
    def test_pcm_sample_rate_is_public(self):
        self.assertEqual(SAMPLE_RATE, 16000)

    def test_iterator_is_lazy_and_stops_between_chunks(self):
        model = FakeModel()
        audio = np.zeros(800, dtype=np.float32)

        with (
            patch("gigaam_mlx.transcribe.split_audio", return_value=CHUNKS),
            patch(
                "gigaam_mlx.transcribe.compute_mel",
                return_value=np.zeros((2, 64), dtype=np.float32),
            ),
        ):
            decoded = iter_decode_audio(audio, model)
            self.assertEqual(model.encode_calls, 0)

            first = next(decoded)
            self.assertEqual(model.encode_calls, 1)
            self.assertEqual(
                first,
                DecodedChunk(
                    start=0.0,
                    end=0.025,
                    emissions=[(10, 0), (11, 1)],
                ),
            )
            decoded.close()
            self.assertEqual(model.encode_calls, 1)

    def test_transcribe_audio_builds_the_existing_segment_shape(self):
        model = FakeModel()
        audio = np.zeros(800, dtype=np.float32)

        with (
            patch("gigaam_mlx.transcribe.split_audio", return_value=CHUNKS),
            patch(
                "gigaam_mlx.transcribe.compute_mel",
                return_value=np.zeros((2, 64), dtype=np.float32),
            ),
        ):
            segments = transcribe_audio(
                audio, model=model, tokenizer=FakeTokenizer(), verbose=False
            )

        self.assertEqual(
            segments,
            [
                {"start": 0.0, "end": 0.025, "text": "first"},
                {"start": 0.025, "end": 0.05, "text": "second"},
            ],
        )

    def test_rejects_non_mono_audio(self):
        with self.assertRaisesRegex(ValueError, "mono"):
            iter_decode_audio(np.zeros((10, 2)), SimpleNamespace())

    def test_transcribe_file_forwards_daemon_limits(self):
        audio = np.zeros(100, dtype=np.float32)
        segments = [{"start": 0.0, "end": 1.0, "text": "ok"}]
        model = SimpleNamespace()
        tokenizer = object()

        with (
            patch("gigaam_mlx.transcribe.load_audio", return_value=audio) as load,
            patch(
                "gigaam_mlx.transcribe.transcribe_audio", return_value=segments
            ) as transcribe,
        ):
            actual = transcribe_file(
                "request.m4a",
                model=model,
                tokenizer=tokenizer,
                verbose=False,
                cache_limit=None,
                load_timeout_s=10,
                max_duration_s=20,
                max_input_bytes=30,
            )

        self.assertEqual(actual, segments)
        load.assert_called_once_with(
            "request.m4a",
            timeout_s=10,
            max_duration_s=20,
            max_input_bytes=30,
        )
        transcribe.assert_called_once_with(
            audio, model=model, tokenizer=tokenizer, verbose=False
        )


if __name__ == "__main__":
    unittest.main()
