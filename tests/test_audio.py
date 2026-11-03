import unittest
import numpy as np
from fill_form.audio import AudioIngress


class AudioTests(unittest.TestCase):
    def test_chunks_flush_once(self):
        a = AudioIngress()
        frames = []
        for _ in range(10):
            frames += a.push(48000, np.ones((4800, 2), dtype=np.int16) * 16384)
        frames += a.finish()
        self.assertEqual(len(frames), 32)
        self.assertEqual(a.finish(), [])
        self.assertAlmostEqual(float(frames[5].mean()), 0.5, places=3)

    def test_rate_change_rejected(self):
        a = AudioIngress()
        a.push(48000, np.zeros(100))
        with self.assertRaises(ValueError):
            a.push(44100, np.zeros(100))
