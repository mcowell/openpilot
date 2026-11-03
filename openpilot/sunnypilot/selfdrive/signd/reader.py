"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Decides, frame by frame, what detector work to run: a full scan of the ROI now and then to find signs, and between
scans, short bursts of cheap reads (local detector + classifier) of each sign found, while it's close enough to read.
Signs are often readable for only ~1 s, so a fixed slow scan misses them, while a fixed fast one keeps the GPU busy for
nothing most of the time. Shared by signd and tools/sign_eval.py so offline results match the car.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from openpilot.sunnypilot.selfdrive.signd.detector import Candidate
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection


class Detector(Protocol):
  """What the reader needs from detector.SignDetector"""
  local: Any  # the local detector model, None if not available

  def scan(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int) -> list[Candidate]: ...
  def refine(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> Candidate | None: ...
  def to_detection(self, c: Candidate, w: int, h: int) -> Detection | None: ...


@dataclass
class ReaderParams:
  scan_hz: float = 1.  # full-ROI detector runs per second
  track_hz: float = 4.  # local reads per second while tracking (one track per read, round-robin)
  track_time: float = 2.  # s, how long a sign found by a scan is followed
  max_tracks: int = 2
  max_misses: int = 2  # consecutive reads that lose the sign, or can't read it, before the track is dropped
  match_iou: float = 0.1  # a scan candidate overlapping a track this much is the same sign
  fallback_scan_hz: float = 2.  # without a local detector there is no tracking: fixed full scans at this rate instead


@dataclass
class Track:
  box: np.ndarray  # x0, y0, x1, y1, ROI pixels, at t
  t: float
  start_t: float
  vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # centre x, centre y (px/s), log height (1/s)
  misses: int = 0

  def predict(self, t: float) -> np.ndarray:
    dt = t - self.t
    cx, cy = (self.box[0] + self.box[2]) / 2 + self.vel[0] * dt, (self.box[1] + self.box[3]) / 2 + self.vel[1] * dt
    k = math.exp(self.vel[2] * dt)
    hw, hh = (self.box[2] - self.box[0]) / 2 * k, (self.box[3] - self.box[1]) / 2 * k
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])

  def update(self, t: float, box: np.ndarray) -> None:
    dt = t - self.t
    if dt > 0:
      old_c, new_c = (self.box[:2] + self.box[2:]) / 2, (box[:2] + box[2:]) / 2
      old_h, new_h = max(1., self.box[3] - self.box[1]), max(1., box[3] - box[1])
      v = np.array([*((new_c - old_c) / dt), math.log(new_h / old_h) / dt])
      self.vel = 0.5 * self.vel + 0.5 * v if self.t > self.start_t else v
    self.box, self.t = box.astype(float), t


def iou(a, b) -> float:
  ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
  iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
  inter = ix * iy
  return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class SignReader:
  def __init__(self, detector: Detector, params: ReaderParams | None = None):
    self.detector = detector
    self.params = params or ReaderParams()
    self.tracks: list[Track] = []
    self.last_scan = -math.inf
    self.last_read = -math.inf
    self._next = 0  # round-robin track index
    self.runs = {'scan': 0, 'read': 0}

  def update(self, t: float, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int) -> list[Detection] | None:
    """Call with every camera frame. Returns detections when detector work ran on this frame, else None."""
    p = self.params
    self.tracks = [tr for tr in self.tracks if t - tr.start_t <= p.track_time and tr.misses <= p.max_misses]

    local = self.detector.local is not None
    if t - self.last_scan >= 1. / (p.scan_hz if local else p.fallback_scan_hz):
      self.last_scan = self.last_read = t
      self.runs['scan'] += 1
      found = self.detector.scan(data, stride, uv_offset, w, h)
      if local:
        self._match(t, found)
      return self._detections(found, w, h)

    if local and self.tracks and t - self.last_read >= 1. / p.track_hz:
      self.last_read = t
      self.runs['read'] += 1
      self._next %= len(self.tracks)
      tr = self.tracks[self._next]
      self._next += 1
      c = self.detector.refine(data, stride, uv_offset, w, h, tr.predict(t))
      if c is None:
        tr.misses += 1
        return []
      tr.update(t, np.array(c.box))
      # found but unreadable also counts against the track, so a non-sign (e.g. a licence plate) doesn't hold the GPU
      tr.misses = 0 if c.value is not None else tr.misses + 1
      return self._detections([c], w, h)

    return None

  def _match(self, t: float, found: list[Candidate]) -> None:
    unmatched, seen = [], set()
    for c in sorted(found, key=lambda c: -c.score):
      box = np.array(c.box, dtype=float)
      tr = max(self.tracks, key=lambda tr: iou(tr.predict(t), box), default=None)
      if tr is not None and iou(tr.predict(t), box) >= self.params.match_iou:
        seen.add(id(tr))
        tr.update(t, box)
        if c.value is not None:
          tr.misses = 0
      else:
        unmatched.append(box)
    for tr in self.tracks:
      if id(tr) not in seen:
        tr.misses += 1  # a full scan that doesn't see a tracked sign is a miss too
    for box in unmatched:
      if len(self.tracks) < self.params.max_tracks:
        self.tracks.append(Track(box=box, t=t, start_t=t))

  def _detections(self, found: list[Candidate], w: int, h: int) -> list[Detection]:
    return [d for c in found if (d := self.detector.to_detection(c, w, h)) is not None]
