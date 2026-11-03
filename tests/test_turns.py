import unittest
import numpy as np
from fill_form.turns import TurnBuffer, TurnConfig


class TurnTests(unittest.TestCase):
    def test_start_and_silence_use_audio_samples(self):
        turn = TurnBuffer(TurnConfig())
        frame = np.zeros(512, dtype=np.float32)
        for _ in range(4):
            self.assertFalse(turn.feed(frame, 0.9))
        self.assertTrue(turn.feed(frame, 0.9))
        self.assertFalse(turn.feed(frame, 0.0))
        self.assertEqual(turn.silence, 512 / 16000)
        turn.reset()
        self.assertFalse(turn.active)

    def test_buffer_stays_bounded_and_rejects_invalid_samples(self):
        turn = TurnBuffer(TurnConfig(max_turn=1.0))
        frame = np.zeros(512, dtype=np.float32)
        for _ in range(100):
            turn.feed(frame, 0.9)
        self.assertTrue(turn.at_limit)
        self.assertLessEqual(len(turn.snapshot().audio), 16000 + 512)
        with self.assertRaises(ValueError):
            turn.feed(np.array([np.nan]), 0.9)
