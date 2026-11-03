"""Turn boundaries use audio samples, never network or inference completion time."""
from collections import deque
from dataclasses import dataclass
import numpy as np
RATE = 16000
FRAME = 512

@dataclass(frozen=True)
class TurnConfig:
    threshold: float = 0.5
    negative_threshold: float = 0.35
    start_speech: float = 0.16
    pre_roll: float = 0.32
    min_silence: float = 0.8
    max_turn: float = 30.0
    partial_interval: float = 1.0

    def __post_init__(self):
        if not 0 < self.negative_threshold < self.threshold < 1:
            raise ValueError('Require 0 < negative_threshold < threshold < 1')
        if not 0 < self.min_silence < self.max_turn:
            raise ValueError('Require 0 < min_silence < max_turn')
        if not 0 < self.start_speech <= self.pre_roll < self.max_turn:
            raise ValueError('Require 0 < start_speech <= pre_roll < max_turn')
        if self.partial_interval <= 0:
            raise ValueError('partial_interval must be positive')

@dataclass(frozen=True)
class Snapshot:
    turn_id: int
    revision: int
    audio: np.ndarray

class TurnBuffer:

    def __init__(self, config):
        self.config = config
        self.clock = 0
        self.turn_id = 0
        self.pre = deque(maxlen=max(1, int(np.ceil(config.pre_roll * RATE / FRAME))))
        self.reset()

    def reset(self):
        self.active = False
        self.frames = []
        self.pre.clear()
        self.run = 0
        self.voicing = False
        self.revision = 0
        self.text_revision = -1
        self.text = ''
        self.started = self.last_voice = self.clock
        self.last_request = -float('inf')

    @property
    def silence(self):
        return (self.clock - self.last_voice) / RATE

    @property
    def at_limit(self):
        return self.active and len(self.frames) * FRAME >= self.config.max_turn * RATE

    def feed(self, frame, probability):
        if frame.shape != (FRAME,) or not np.isfinite(frame).all():
            raise ValueError('Expected 512 finite mono samples at 16 kHz')
        self.clock += FRAME
        threshold = self.config.negative_threshold if self.voicing else self.config.threshold
        self.voicing = probability >= threshold
        self.run = self.run + FRAME if self.voicing else 0
        started = False
        if not self.active:
            self.pre.append(frame.copy())
            if self.run >= self.config.start_speech * RATE:
                self.turn_id += 1
                self.active = True
                started = True
                self.frames = list(self.pre)
                self.started = self.clock - len(self.frames) * FRAME
                self.pre.clear()
        elif not self.at_limit:
            self.frames.append(frame.copy())
        if self.active and self.voicing and (not self.at_limit or started):
            self.last_voice = self.clock
            self.revision += 1
        return started

    def needs_asr(self):
        return self.active and self.revision != self.text_revision and (self.at_limit or self.silence >= self.config.min_silence / 2 or self.clock / RATE - self.last_request >= self.config.partial_interval)

    def snapshot(self):
        self.last_request = self.clock / RATE
        return Snapshot(self.turn_id, self.revision, np.concatenate(self.frames).copy())

    def set_text(self, turn_id, revision, text):
        if not self.active or turn_id != self.turn_id or revision < self.text_revision:
            return False
        self.text_revision, self.text = (revision, text.strip())
        return True
