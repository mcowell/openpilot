"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Temporal filter that turns noisy per-frame sign detections into a stable speed limit.
Shared by the on-device daemon (signd) and the offline evaluation tool (tools/sign_eval.py),
so offline scores predict on-device behaviour.
"""
import math
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
  value: int  # number printed on the sign, in sign units (km/h or mph)
  confidence: float  # 0..1, product of detector and classifier confidence
  box: tuple[float, float, float, float]  # x, y, w, h, normalized to the full frame

  @property
  def area(self) -> float:
    return self.box[2] * self.box[3]


@dataclass
class FilterParams:
  min_confidence: float = 0.6  # per-detection threshold
  min_hits: int = 2  # agreeing detections needed to accept a value (at SIGND_RATE_HZ = 1, a sign is in view for ~2-4 frames)
  window: float = 2.5  # s, detections older than this are dropped (min_hits reads at SIGND_RATE_HZ must fit)
  min_area_growth: float = 1.0  # last/first box area; >= 1 means the sign is being approached, not left behind
  max_hold_time: float = 600.  # s, forget the limit after this long without a new sign
  max_hold_distance: float = 5000.  # m, forget the limit after driving this far without a new sign
  turn_window: float = 10.  # s, window over which heading change is summed
  turn_reset_angle: float = math.radians(60.)  # rad, a turn this sharp means we are likely on a new road


@dataclass
class SignFilterState:
  speed_limit: int = 0  # in sign units, 0 when unknown
  confidence: float = 0.
  time_since_accept: float = 0.
  distance_since_accept: float = 0.


@dataclass
class SignFilter:
  params: FilterParams = field(default_factory=FilterParams)

  def __post_init__(self):
    self._hits: deque[tuple[float, Detection]] = deque()
    self._yaw: deque[tuple[float, float]] = deque()  # (t, heading change during step)
    self._last_t: float | None = None
    self.state = SignFilterState()

  def reset(self) -> None:
    self._hits.clear()
    self._yaw.clear()
    self.state = SignFilterState()

  def update(self, t: float, detections: list[Detection], v_ego: float = 0., yaw_rate: float = 0.) -> SignFilterState:
    dt = 0. if self._last_t is None else max(0., t - self._last_t)
    self._last_t = t

    self._update_hold(t, dt, v_ego, yaw_rate)

    p = self.params
    for det in detections:
      if det.confidence >= p.min_confidence:
        self._hits.append((t, det))
    while self._hits and t - self._hits[0][0] > p.window:
      self._hits.popleft()

    candidate = self._candidate()
    if candidate is not None:
      value, confidence = candidate
      self._hits.clear()
      self._yaw.clear()
      self.state = SignFilterState(speed_limit=value, confidence=confidence)

    return self.state

  def _update_hold(self, t: float, dt: float, v_ego: float, yaw_rate: float) -> None:
    if self.state.speed_limit == 0:
      return

    p = self.params
    self.state.time_since_accept += dt
    self.state.distance_since_accept += max(0., v_ego) * dt

    self._yaw.append((t, yaw_rate * dt))
    while self._yaw and t - self._yaw[0][0] > p.turn_window:
      self._yaw.popleft()
    heading_change = abs(sum(d for _, d in self._yaw))

    if (self.state.time_since_accept > p.max_hold_time or
        self.state.distance_since_accept > p.max_hold_distance or
        heading_change > p.turn_reset_angle):
      self.reset()

  def _candidate(self) -> tuple[int, float] | None:
    by_value: dict[int, list[Detection]] = {}
    for _, det in self._hits:
      by_value.setdefault(det.value, []).append(det)

    best: tuple[int, float] | None = None
    for value, dets in by_value.items():
      if len(dets) < self.params.min_hits:
        continue
      if dets[-1].area < dets[0].area * self.params.min_area_growth:
        continue
      confidence = sum(d.confidence for d in dets) / len(dets)
      if best is None or confidence > best[1]:
        best = (value, confidence)
    return best
