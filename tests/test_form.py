import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

from fill_form.config import Settings
from fill_form.backend import Controller
from fill_form.form import normalize_value


class FormTests(unittest.TestCase):
    def setUp(self):
        self.c = Controller(Settings(fake=True, silence=0.7))
        self.sid = self.c.dispatch(None, "open")

    def tearDown(self):
        self.c.close()

    def snapshot(self):
        return self.c.dispatch(self.sid, "snapshot")["form"]

    def wait(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.snapshot()
            if predicate(result):
                return result
            time.sleep(0.02)
        self.fail(str(result))

    def test_dictation_appends_without_llm_and_sessions_are_isolated(self):
        self.c.dispatch(self.sid, "text", "Nguyễn Văn")
        self.wait(lambda s: s["values"]["full_name"] == "Nguyễn Văn")
        self.c.dispatch(self.sid, "text", "Nam")
        self.wait(lambda s: s["values"]["full_name"] == "Nguyễn Văn Nam")
        other = self.c.dispatch(None, "open")
        self.assertEqual(self.c.dispatch(other, "snapshot")["form"]["values"]["full_name"], "")

    def test_manual_edits_clear_and_export_validation(self):
        self.c.dispatch(self.sid, "edit_field", "full_name", "Nam")
        self.c.dispatch(self.sid, "edit_field", "phone", "0901234567")
        self.c.dispatch(self.sid, "edit_field", "address", "Hà Nội")
        data = json.loads(self.c.dispatch(self.sid, "export_form"))
        self.assertEqual(data, {"full_name": "Nam", "phone": "0901234567", "address": "Hà Nội"})
        self.c.dispatch(self.sid, "edit_field", "phone", "")
        with self.assertRaises(ValueError):
            self.c.dispatch(self.sid, "export_form")
        with self.assertRaises(ValueError):
            self.c.dispatch(self.sid, "select_field", "unknown")

    def test_only_one_session_can_use_microphone(self):
        other = self.c.dispatch(None, "open")
        self.c.dispatch(self.sid, "mic", True)
        with self.assertRaises(ValueError):
            self.c.dispatch(other, "mic", True)
        self.c.dispatch(self.sid, "mic", False)
        self.c.dispatch(other, "mic", True)
        self.assertTrue(self.c.dispatch(other, "snapshot")["mic"])

    def test_invalid_phone_preserves_previous_value(self):
        self.c.dispatch(self.sid, "edit_field", "phone", "090")
        self.c.dispatch(self.sid, "select_field", "phone")
        self.c.dispatch(self.sid, "text", "tôi không biết")
        result = self.wait(lambda s: bool(s["error"]))
        self.assertEqual(result["values"]["phone"], "090")
        self.assertEqual(result["transcript"], "tôi không biết")

    def test_pending_audio_keeps_original_field_and_manual_edit_wins(self):
        async def speech():
            client = self.c.clients[self.sid]
            client.runner.turn.turn_id = 1
            client.session.speech_started()
            return client.session.generation

        generation = self.c.call(speech())
        self.c.dispatch(self.sid, "select_field", "address")

        async def finish():
            self.c.clients[self.sid].session.submit("Nam", generation, 1)

        self.c.call(finish())
        self.assertEqual(self.snapshot()["values"]["full_name"], "Nam")
        self.assertEqual(self.snapshot()["values"]["address"], "")
        generation = self.c.call(speech())
        self.c.dispatch(self.sid, "edit_field", "address", "Đà Nẵng")
        self.c.call(finish())
        self.assertEqual(self.snapshot()["values"]["address"], "Đà Nẵng")
        self.assertTrue(self.snapshot()["error"])

    def test_audio_through_vad_updates_form(self):
        self.c.pool.recognize = lambda audio: ("Nguyễn Văn Nam", 0.02)
        self.c.dispatch(self.sid, "mic", True)
        self.c.dispatch(self.sid, "audio", (16000, np.ones(8192, np.float32) * 0.1))
        self.c.dispatch(self.sid, "mic", False)
        self.wait(lambda s: s["values"]["full_name"] == "Nguyễn Văn Nam")
        self.assertTrue(any(e["event"] == "segment_closed" for e in self.c.dispatch(self.sid, "snapshot")["events"]))

    def test_asr_failure_does_not_erase_form(self):
        def fail(audio):
            raise RuntimeError("recognition failed")
        self.c.pool.recognize = fail
        self.c.dispatch(self.sid, "edit_field", "full_name", "Nam")
        self.c.dispatch(self.sid, "mic", True)
        self.c.dispatch(self.sid, "audio", (16000, np.ones(8192, np.float32) * 0.1))
        self.c.dispatch(self.sid, "mic", False)
        self.wait(lambda s: bool(s["error"]))
        self.assertEqual(self.snapshot()["values"]["full_name"], "Nam")

    def test_export_waits_for_microphone_and_pending_recognition(self):
        self.c.dispatch(self.sid, "edit_field", "full_name", "Nam")
        self.c.dispatch(self.sid, "edit_field", "phone", "0901234567")
        self.c.dispatch(self.sid, "edit_field", "address", "Hà Nội")
        self.c.dispatch(self.sid, "mic", True)
        with self.assertRaisesRegex(ValueError, "mic"):
            self.c.dispatch(self.sid, "export_form")

    def test_delayed_audio_packet_stays_before_field_selection(self):
        self.c.pool.recognize = lambda audio: ("Nam", 0.02)
        def packet(sequence, audio="", mic=True):
            return dict(sequence=sequence, recording_id=1, rate=16000,
                        mic=mic, audio=audio, acks=[])
        audio = base64.b64encode((np.ones(8192, dtype="<i2") * 5000).tobytes()).decode()
        with ThreadPoolExecutor() as executor:
            selected = executor.submit(self.c.dispatch, self.sid, "form_packet",
                                       packet(2), "select_field", "address")
            time.sleep(0.05)
            self.c.dispatch(self.sid, "packet", packet(1, audio))
            selected.result(timeout=5)
        self.c.dispatch(self.sid, "packet", packet(3, mic=False))
        self.wait(lambda s: bool(s["values"]["full_name"]))
        self.assertEqual(self.snapshot()["values"], {"full_name": "Nam", "phone": "", "address": ""})


class ValueTests(unittest.TestCase):
    def test_phone_digits_and_spoken_digits_keep_leading_zero(self):
        self.assertEqual(normalize_value("phone", "KHÔNG chín không, một hai ba bốn năm sáu bảy."), "0901234567")
        self.assertEqual(normalize_value("phone", "090 123-4567"), "0901234567")
        self.assertEqual(normalize_value("phone", "+84 90 123 4567"), "+84901234567")
        with self.assertRaises(ValueError):
            normalize_value("phone", "chín mươi")

    def test_form_config_needs_no_llm_credentials(self):
        with patch.dict(os.environ, {"ASR_ENV_PATH": "/nonexistent"}, clear=True):
            self.assertFalse(Settings.load().fake)
