#!/usr/bin/env python3
"""
Lightweight continuous audio stream with per-sample easing and a brief startup delay.

- Background thread generates a stereo sine wave continuously
- Targets (freq/amp/pan) are updated via enqueue_tone, then eased smoothly
- Small initial delay gives time for error correction and smoothing to stabilize
- Moderate sample rate and short frames to keep CPU overhead low

If simpleaudio is not available, all methods become no-ops.
"""
from __future__ import annotations

import math
import threading
import queue
import time
from typing import Optional

import numpy as np

try:
    import simpleaudio as sa  # type: ignore
except Exception:
    sa = None


class AudioStreamer:
    def __init__(
        self,
        sample_rate: int = 12000,
        frame_sec: float = 0.1,
        smoothing_ms: float = 50.0,
        initial_delay_sec: float = 0.2,
        max_events: int = 64,
    ):
        """
        sample_rate: moderate SR to be gentle on CPU
        frame_sec:   small frame for low-latency updates (~100 ms)
        smoothing_ms: one-pole smoothing time constant for freq/amp/pan
        initial_delay_sec: small startup delay before playback begins
        """
        self.sr = int(sample_rate)
        self.N = max(32, int(self.sr * float(frame_sec)))
        self.dt = 1.0 / float(self.sr)
        self.tau = max(1e-3, smoothing_ms / 1000.0)
        # per-sample alpha for exponential smoothing
        self.alpha = 1.0 - math.exp(-self.dt / self.tau)

        self.initial_delay_sec = max(0.0, float(initial_delay_sec))

        self.events: "queue.Queue[tuple[float, float, float]]" = queue.Queue(maxsize=max_events)

        # smoothed state
        self._freq = 440.0
        self._amp = 0.0
        self._pan = 0.0
        # targets
        self._tfreq = 440.0
        self._tamp = 0.0
        self._tpan = 0.0

        # phase accumulator for sine
        self._phase = 0.0

        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self):
        if sa is None:
            return
        if self._th is not None and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            try:
                self._th.join(timeout=0.5)
            except RuntimeError:
                pass

    def enqueue_tone(self, freq_hz: float, amp: float = 0.2, pan: float = 0.0):
        """
        Update target frequency, amplitude, and pan. The generator eases to these targets.
        - freq_hz:    target frequency in Hz
        - amp:        [0..1], target amplitude
        - pan:        [-1..1], -1 left, +1 right
        """
        if sa is None:
            return
        # sanitize targets
        f = float(max(20.0, min(20000.0, freq_hz)))
        a = float(max(0.0, min(1.0, amp)))
        p = float(max(-1.0, min(1.0, pan)))
        if self.events.full():
            try:
                _ = self.events.get_nowait()
            except queue.Empty:
                pass
        try:
            self.events.put_nowait((f, a, p))
        except queue.Full:
            pass

    def _apply_events(self):
        # Drain all pending events and set the latest as target
        while True:
            try:
                f, a, p = self.events.get_nowait()
                self._tfreq, self._tamp, self._tpan = f, a, p
            except queue.Empty:
                break

    def _run(self):
        # Optional startup delay
        if self.initial_delay_sec > 0.0:
            time.sleep(self.initial_delay_sec)

        # initialize smoothed state at targets (avoids cold-start pop)
        self._freq = self._tfreq
        self._amp = self._tamp
        self._pan = self._tpan

        while not self._stop.is_set():
            # Accept any new target updates
            self._apply_events()

            # Generate one frame with easing toward targets
            # Prepare buffers
            y = np.empty(self.N, dtype=np.float32)
            freq = self._freq
            amp = self._amp
            pan = self._pan
            tfreq = self._tfreq
            tamp = self._tamp
            tpan = self._tpan
            alpha = self.alpha
            phase = self._phase
            sr = float(self.sr)

            for i in range(self.N):
                # exponential smoothing per sample
                freq += (tfreq - freq) * alpha
                amp += (tamp - amp) * alpha
                pan += (tpan - pan) * alpha

                phase += 2.0 * math.pi * (freq / sr)
                # keep phase bounded to avoid float drift
                if phase > 1e12:
                    phase = math.fmod(phase, 2.0 * math.pi)

                y[i] = math.sin(phase) * amp

            # commit smoothed state back
            self._freq, self._amp, self._pan = freq, amp, pan
            self._phase = phase

            # stereo pan
            l = float(np.clip(0.5 * (1.0 - pan), 0.0, 1.0))
            r = float(np.clip(0.5 * (1.0 + pan), 0.0, 1.0))
            stereo = np.stack([y * l, y * r], axis=1)

            # convert to int16 PCM and play
            try:
                pcm = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
                sa.play_buffer(pcm, num_channels=2, bytes_per_sample=2, sample_rate=self.sr)
            except Exception:
                # If playback fails transiently, don’t spin
                time.sleep(0.01)
                continue

            # tiny breather to avoid hammering the CPU in rare cases
            time.sleep(0.001)