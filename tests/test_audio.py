import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

from gigaam_mlx.audio import compute_mel, load_audio, split_audio


def reference_split_audio(audio, max_chunk_sec=20.0, sr=16000):
    chunk_samples = int(max_chunk_sec * sr)
    min_silence = int(0.3 * sr)
    chunks = []
    start = 0

    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        if end < len(audio):
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


class SplitAudioTest(unittest.TestCase):
    def test_matches_the_previous_sliding_energy_algorithm(self):
        rng = np.random.default_rng(7)
        audio = rng.uniform(-0.5, 0.5, 4200).astype(np.float32)
        audio[650:1050] = 0.0
        audio[1900:2350] = 0.0
        audio[3150:3600] = 0.0
        expected = reference_split_audio(audio, max_chunk_sec=1.2, sr=1000)
        original_convolve = np.convolve

        with patch("numpy.convolve", wraps=original_convolve) as convolve:
            actual = split_audio(audio, max_chunk_sec=1.2, sr=1000)

        self.assertEqual(actual, expected)
        for call in convolve.call_args_list:
            self.assertLessEqual(len(call.args[0]), int(0.3 * 1000) + 16)

    def test_rejects_non_positive_chunk_sizes(self):
        with self.assertRaisesRegex(ValueError, "chunk duration"):
            split_audio(np.zeros(10), max_chunk_sec=0)


class MelTest(unittest.TestCase):
    def test_pads_audio_shorter_than_one_analysis_window(self):
        mel = compute_mel(np.zeros(10, dtype=np.float32))
        self.assertEqual(mel.shape, (1, 64))


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
class LoadAudioTest(unittest.TestCase):
    def make_wav(self, duration_s=0.1, sr=16000):
        handle = tempfile.NamedTemporaryFile(suffix=".wav")
        samples = np.zeros(round(duration_s * sr), dtype=np.int16)
        with wave.open(handle.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(samples.tobytes())
        return handle

    def test_enforces_maximum_input_size_before_starting_ffmpeg(self):
        with self.make_wav() as source:
            with self.assertRaisesRegex(ValueError, "input size"):
                load_audio(source.name, max_input_bytes=1)

    def test_streams_pcm_with_a_maximum_duration(self):
        with self.make_wav() as source:
            audio = load_audio(source.name, max_duration_s=0.2, timeout_s=5)
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(audio.shape, (1600,))

            with self.assertRaisesRegex(ValueError, "maximum duration"):
                load_audio(source.name, max_duration_s=0.05, timeout_s=5)

    def test_reports_ffmpeg_errors_without_hanging(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as source:
            source.write(b"not an audio file")
            source.flush()
            with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                load_audio(source.name, timeout_s=5)

    def test_timeout_reaps_ffmpeg_process(self):
        real_popen = subprocess.Popen
        processes = []

        def start_hanging_process(*args, **kwargs):
            process = real_popen(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
            )
            processes.append(process)
            return process

        with patch(
            "gigaam_mlx.audio.subprocess.Popen", side_effect=start_hanging_process
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                load_audio("unused", timeout_s=0.01)

        self.assertIsNotNone(processes[0].poll())

    def test_timeout_applies_while_ffmpeg_is_continuously_writable(self):
        real_popen = subprocess.Popen
        processes = []

        def start_writing_process(*args, **kwargs):
            process = real_popen(
                [
                    sys.executable,
                    "-c",
                    "import os\nwhile True: os.write(1, b'0' * 65536)",
                ],
                stdout=kwargs["stdout"],
                stderr=kwargs["stderr"],
            )
            processes.append(process)
            return process

        with patch(
            "gigaam_mlx.audio.subprocess.Popen", side_effect=start_writing_process
        ):
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                load_audio("unused", timeout_s=0.01)

        self.assertIsNotNone(processes[0].poll())
