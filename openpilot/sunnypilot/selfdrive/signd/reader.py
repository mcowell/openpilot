"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Schedules signd's GPU work as small jobs, each short enough (~10-15 ms on a 3X) to fit in the gap after the driving model
finishes a frame, so signd never holds the GPU when modeld needs it:
  tile:     the local detector on one tile of the ROI; a sweep over all tiles is a scan, and finds new signs
  local:    the local detector in a window around a tracked sign's predicted position, to re-find it
  classify: the classifier on a crop cut (on the CPU) from the frame where a sign was found
Signs are often readable for only ~1 s, so a found sign is read several times a second for a couple of seconds.
prepare() picks the next job and does its CPU work (while the GPU is busy with modeld); run() does the GPU part.
Shared by signd and tools/sign_eval.py so offline results match the car.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from openpilot.sunnypilot.selfdrive.signd.detector import Candidate, Window, best_overlap
from openpilot.sunnypilot.selfdrive.signd.guard import GuardLevel
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection


class Detector(Protocol):
  """What the reader needs from detector.SignDetector"""
  local: Any  # the local detector model, None if not available

  def scan(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int) -> list[Candidate]: ...
  def tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]: ...
  def entry_tiles(self, w: int, h: int) -> list[tuple[int, int, int, int]]: ...
  def prepare_tile(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, rect) -> Window: ...
  def prepare_local(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> Window: ...
  def run_window(self, win: Window, w: int, h: int) -> np.ndarray: ...
  def prepare_crop(self, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int, box) -> np.ndarray | None: ...
  def run_classifier(self, x: np.ndarray) -> tuple[int | None, float]: ...
  def candidate(self, box, score: float, value: int | None, prob: float) -> Candidate: ...
  def to_detection(self, c: Candidate, w: int, h: int) -> Detection | None: ...


@dataclass
class ReaderParams:
  job_every: int = 2  # one GPU job per this many driving-model frames (20 Hz); doubled when the guard says reduced
  job_every_tracking: int = 1  # ... while a sign is being tracked: signs are often readable for only ~1 s
  track_hz: float = 5.  # local reads per second of each tracked sign (each also needs a classify job)
  track_time: float = 2.  # s, how long a found sign is followed
  max_tracks: int = 2
  max_misses: int = 2  # consecutive reads that lose the sign, or can't read it, before the track is dropped
  match_iou: float = 0.1  # a tile detection overlapping a track this much is the same sign
  max_pending: int = 3  # crops waiting for the classifier; the oldest are dropped beyond this
  ignore_time: float = 3.  # s a dropped track's area is ignored by tiles, so e.g. a licence plate isn't re-tracked at once
  fallback_scan_hz: float = 2.  # without a local detector: fixed full scans (one long job) at this rate instead


@dataclass
class Track:
  box: np.ndarray  # x0, y0, x1, y1, ROI pixels, at t
  t: float
  start_t: float
  vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # centre x, centre y (px/s), log height (1/s)
  misses: int = 0
  last_read: float = -math.inf

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


@dataclass
class Job:
  kind: str  # 'tile', 'local', 'classify' or 'scan'
  t: float
  frame: tuple[np.ndarray, int, int, int, int]  # data, stride, uv_offset, w, h: crops are cut from it after run()
  window: Window | None = None
  box: np.ndarray | None = None  # local: predicted box; classify: box the crop was cut around
  score: float = 0.
  crop: np.ndarray | None = None
  track: Track | None = None


class SignReader:
  def __init__(self, detector: Detector, params: ReaderParams | None = None):
    self.detector = detector
    self.params = params or ReaderParams()
    self.tracks: list[Track] = []
    self.pending: list[Job] = []  # classify jobs
    self.ignored: list[tuple[np.ndarray, float]] = []  # (box, until) of tracks dropped for repeated misses
    self.last_scan = -math.inf
    self._slot = 0
    self._tile = 0
    self.runs = {'scan': 0, 'tile': 0, 'local': 0, 'classify': 0}

  def update(self, t: float, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int,
             level: GuardLevel = GuardLevel.normal) -> list[Detection] | None:
    """prepare() + run(), for offline use. Returns detections when a job ran on this frame, else None."""
    job = self.prepare(t, data, stride, uv_offset, w, h, level)
    return None if job is None else self.run(job)

  def prepare(self, t: float, data: np.ndarray, stride: int, uv_offset: int, w: int, h: int,
              level: GuardLevel = GuardLevel.normal) -> Job | None:
    """Call with every camera frame. Picks this frame's job, if any, and does its CPU work.
    level (guard.ModelGuard): reduced runs half as many jobs and doesn't track, paused runs none."""
    p = self.params
    lost = [tr for tr in self.tracks if tr.misses > p.max_misses]
    self.ignored = [(b, until) for b, until in self.ignored if until > t] + [(tr.box, t + p.ignore_time) for tr in lost]
    self.tracks = [tr for tr in self.tracks if t - tr.start_t <= p.track_time and tr.misses <= p.max_misses]
    if level == GuardLevel.paused:
      return None
    frame = (data, stride, uv_offset, w, h)

    if self.detector.local is None:
      scan_hz = p.fallback_scan_hz / (2 if level == GuardLevel.reduced else 1)
      if t - self.last_scan < 1. / scan_hz:
        return None
      self.last_scan = t
      return Job('scan', t, frame)

    self._slot += 1
    if level == GuardLevel.reduced:
      every = 2 * p.job_every
    else:
      every = p.job_every_tracking if self.tracks or self.pending else p.job_every
    if (self._slot - 1) % every:
      return None

    if self.pending:
      return self.pending.pop(0)
    if level == GuardLevel.normal:
      due = [tr for tr in self.tracks if t - tr.last_read >= 1. / p.track_hz]
      if due:
        tr = min(due, key=lambda tr: tr.last_read)
        tr.last_read = t
        box = tr.predict(t)
        return Job('local', t, frame, window=self.detector.prepare_local(*frame, box), box=box, track=tr)
    # alternate the entry tiles (where new signs appear) with the full grid, so new signs are found within ~0.4 s
    entry, grid = self.detector.entry_tiles(w, h), self.detector.tiles(w, h)
    n = self._tile
    self._tile += 1
    rect = entry[(n // 2) % len(entry)] if n % 2 == 0 else grid[(n // 2) % len(grid)]
    return Job('tile', t, frame, window=self.detector.prepare_tile(*frame, rect))

  def run(self, job: Job) -> list[Detection]:
    """GPU part of a job, plus bookkeeping. Returns the readings it produced (only classify and scan jobs read)."""
    self.runs[job.kind] += 1
    w, h = job.frame[3], job.frame[4]

    if job.kind == 'scan':
      return [d for c in self.detector.scan(*job.frame) if (d := self.detector.to_detection(c, w, h)) is not None]

    if job.kind == 'classify':
      assert job.crop is not None and job.box is not None
      c = self.detector.candidate(job.box, job.score, *self.detector.run_classifier(job.crop))
      if job.track is not None:
        # unreadable counts against the track, so a non-sign (e.g. a licence plate) doesn't keep the GPU busy
        job.track.misses = 0 if c.value is not None else job.track.misses + 1
      d = self.detector.to_detection(c, w, h)
      return [] if d is None else [d]

    assert job.window is not None
    found = self.detector.run_window(job.window, w, h)
    if job.kind == 'local':
      assert job.track is not None and job.box is not None
      best = best_overlap(found, job.box)
      if best is None:
        job.track.misses += 1
      else:
        job.track.update(job.t, best[:4])
        self._queue(job, best[:4], float(best[4]), job.track)
      return []

    # tile: new signs start tracks, whose first job is a local read (the tile's box is coarse, so it would make a poor
    # classifier crop); signs already tracked just update their position
    for row in found[np.argsort(-found[:, 4])]:
      box = row[:4]
      tr = max(self.tracks, key=lambda tr: iou(tr.predict(job.t), box), default=None)
      if tr is not None and iou(tr.predict(job.t), box) >= self.params.match_iou:
        tr.update(job.t, box)
      elif len(self.tracks) < self.params.max_tracks and not any(iou(b, box) >= self.params.match_iou for b, _ in self.ignored):
        self.tracks.append(Track(box=box.astype(float), t=job.t, start_t=job.t))
    return []

  def _queue(self, job: Job, box: np.ndarray, score: float, track: Track) -> None:
    crop = self.detector.prepare_crop(*job.frame, box)
    if crop is None:
      return
    self.pending.append(Job('classify', job.t, job.frame, box=box, score=score, crop=crop, track=track))
    del self.pending[:-self.params.max_pending]
