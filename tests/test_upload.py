from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf

from fill_form.backend import Controller
from fill_form.config import Settings
from fill_form.audio import load_audio_file


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "speech.wav"
        self.c = Controller(Settings(fake=True, silence=0.7))
        self.sid = self.c.dispatch(None, "open")

    def tearDown(self):
        self.c.close()
        self.folder.cleanup()

    def wait(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            snapshot = self.c.dispatch(self.sid, "snapshot")
            if predicate(snapshot):
                return snapshot
            time.sleep(0.02)
        self.fail(str(snapshot))

    def test_upload_resamples_stereo_and_runs_vad_for_each_segment(self):
        rate = 48000
        audio = np.concatenate((np.ones(rate // 2) * 0.2, np.zeros(rate), np.ones(rate // 2) * 0.2))
        sf.write(self.path, np.column_stack((audio, audio)), rate)
        self.c.pool.recognize = lambda audio: ("Hà Nội", 0.02)
        self.c.dispatch(self.sid, "select_field", "address")
        self.c.dispatch(self.sid, "audio_file", str(self.path), False)
        result = self.wait(lambda s: not s["file_active"])
        self.assertEqual(result["form"]["values"]["address"], "Hà Nội Hà Nội")
        self.assertEqual(sum(e["event"] == "segment_closed" for e in result["events"]), 2)
        self.assertFalse(result["mic"])

    def test_file_finishes_when_vad_has_speech_hangover(self):
        sf.write(self.path, np.ones(8192) * 0.2, 16000)
        remaining = 0
        def vad(frame):
            nonlocal remaining
            if np.max(np.abs(frame)) > 0.01:
                remaining = 6
                return 0.9
            if remaining:
                remaining -= 1
                return 0.9
            return 0.0
        self.c.clients[self.sid].runner.vad = vad
        self.c.pool.recognize = lambda audio: ("Nam", 0.02)
        self.c.dispatch(self.sid, "audio_file", str(self.path), False)
        result = self.wait(lambda s: not s["file_active"])
        self.assertEqual(result["form"]["values"]["full_name"], "Nam")

    def test_upload_keeps_original_field_and_manual_edit_wins(self):
        sf.write(self.path, np.ones(16000) * 0.2, 16000)
        started, release = threading.Event(), threading.Event()
        def recognize(audio):
            started.set()
            release.wait(5)
            return "Nam", 0.02
        self.c.pool.recognize = recognize
        self.c.dispatch(self.sid, "audio_file", str(self.path), False)
        try:
            self.assertTrue(started.wait(3))
            self.c.dispatch(self.sid, "select_field", "address")
            self.c.dispatch(self.sid, "edit_field", "full_name", "Nguyễn Văn Nam")
            with self.assertRaises(ValueError):
                self.c.dispatch(self.sid, "mic", True)
            with self.assertRaises(ValueError):
                self.c.dispatch(self.sid, "audio_file", str(self.path), False)
            with self.assertRaises(ValueError):
                self.c.dispatch(self.sid, "export_form")
        finally:
            release.set()
        result = self.wait(lambda s: not s["file_active"])
        self.assertEqual(result["form"]["values"]["full_name"], "Nguyễn Văn Nam")
        self.assertEqual(result["form"]["values"]["address"], "")

    def test_silence_and_invalid_file_preserve_data(self):
        self.c.dispatch(self.sid, "edit_field", "full_name", "Nam")
        sf.write(self.path, np.zeros(16000), 16000)
        self.c.dispatch(self.sid, "audio_file", str(self.path), False)
        result = self.wait(lambda s: not s["file_active"])
        self.assertEqual(result["form"]["values"]["full_name"], "Nam")
        self.path.write_bytes(b"not audio")
        self.c.dispatch(self.sid, "audio_file", str(self.path), False)
        result = self.wait(lambda s: not s["file_active"])
        self.assertTrue(result["form"]["error"])
        self.assertEqual(result["form"]["values"]["full_name"], "Nam")

    def test_rejects_empty_or_overlong_audio(self):
        sf.write(self.path, np.empty(0), 16000)
        with self.assertRaises(ValueError):
            load_audio_file(str(self.path))
        sf.write(self.path, np.zeros(8000 * 301, dtype=np.float32), 8000)
        with self.assertRaisesRegex(ValueError, "5 phút"):
            load_audio_file(str(self.path))

    def test_ffmpeg_fallback_decodes_to_finite_16khz_frames(self):
        self.path.write_bytes(b"encoded audio")
        result = SimpleNamespace(returncode=0, stdout=(np.ones(16000, dtype="<f4") * 0.2).tobytes())
        with patch("fill_form.audio.sf.SoundFile", side_effect=RuntimeError), \
             patch("fill_form.audio.shutil.which", return_value="ffmpeg"), \
             patch("fill_form.audio.subprocess.run", return_value=result):
            frames, duration, rate = load_audio_file(str(self.path))
        self.assertEqual((duration, rate), (1.0, 16000))
        self.assertEqual(frames[0].shape, (512,))
        np.testing.assert_allclose(frames[0], 0.2)
        self.assertTrue(all(np.isfinite(frame).all() for frame in frames))
