"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Temporal filter that turns noisy per-frame sign detections into a stable speed limit.
Shared by the on-device daemon (signd) and the offline evaluation tool (tools/sign_eval.py),
so offline scores predict on-device behaviour.
"""
import math
import re
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
  value: int  # number printed on the sign, in sign units (km/h or mph)
  confidence: float  # 0..1, product of detector and classifier confidence
  box: tuple[float, float, float, float]  # x, y, w, h, normalized to the full frame
  school: float = 0.  # probability of a school zone pentagon on the sign

  @property
  def area(self) -> float:
    return self.box[2] * self.box[3]

  @property
  def clipped_v(self) -> bool:
    """box touches the top or bottom of the frame, so its height is cut off"""
    return self.box[1] <= EDGE or self.box[1] + self.box[3] >= 1. - EDGE

  @property
  def clipped_h(self) -> bool:
    return self.box[0] <= EDGE or self.box[0] + self.box[2] >= 1. - EDGE


EDGE = 0.005  # normalized; a box this close to the frame edge is cut off by it
METRIC_STEP = 10  # km/h limits are posted in steps of 10 (Canada); a 15 or 45 is a misread (often a county road marker)
ROAD_NUMBER_RE = re.compile(r'\b(\d{1,3})\b')


def plausible(value: int, is_metric: bool, road: str) -> bool:
  """Whether a sign reading can be a posted limit here. 0 (end of zone) always is."""
  if value == 0:
    return True
  if is_metric and value % METRIC_STEP:
    return False
  # a route marker for the road being driven ("County Road 15" -> a "15" shield) reads like a limit sign
  return str(value) not in ROAD_NUMBER_RE.findall(road)


def growth(first: Detection, last: Detection) -> float:
  """How much a sign grew between two detections (> 1: being approached). A sign leaving the frame (often over the top
  while passing it, or as the car turns) is cut off by the edge, so its area shrinks although it's still getting closer:
  compare the dimension the edge doesn't cut."""
  if first.clipped_v or last.clipped_v:
    return last.box[2] / max(first.box[2], 1e-6) if not (first.clipped_h or last.clipped_h) else 1.
  if first.clipped_h or last.clipped_h:
    return last.box[3] / max(first.box[3], 1e-6)
  return last.area / max(first.area, 1e-9)


@dataclass
class FilterParams:
  min_confidence: float = 0.6  # per-detection threshold
  min_hits: int = 2  # agreeing detections needed to accept a value (at SIGND_RATE_HZ = 2, a sign is often readable for only ~1 s)
  window: float = 2.5  # s, detections older than this are dropped (min_hits reads at SIGND_RATE_HZ must fit)
  # when the map knows the limit and the sign says otherwise (work zones, school zones, or a misread), want more reads.
  # Equal to min_hits: off
  min_hits_map_disagree: int = 2
  # a sign that agrees with the map needs only this many reads: at signd's rate, a sign passed at speed or just after a
  # turn is often read once. Accepting it can't show anything the map doesn't already say.
  min_hits_map_agree: int = 1
  min_area_growth: float = 1.0  # last/first box size (growth()); >= 1 means the sign is being approached, not left behind
  # a limit holds until the road ends: the road name, when the map knows it, says when that is. Signs can be 8+ km apart.
  max_hold_distance_on_road: float = 20000.  # m, forget the limit after this far on one road without a new sign
  road_change_distance: float = 100.  # m on a new road name before it counts: map matching flickers at intersections
  road_grace_distance: float = 200.  # m; a sign read this soon before the name changes belongs to the new road
  # turning off the mapped road (into a parking lot or driveway): the name goes empty instead of changing
  off_road_distance: float = 15.  # m driven with no road name, after a sharp turn, before the limit is dropped
  # without a road name (no map data), fall back on:
  max_hold_time: float = 600.  # s, forget the limit after this long without a new sign
  max_hold_distance: float = 5000.  # m, forget the limit after driving this far without a new sign
  turn_window: float = 10.  # s, window over which heading change is summed
  turn_reset_angle: float = math.radians(60.)  # rad, a turn this sharp means we are likely on a new road (with a road
  # name: only if the name changed too, as a road can bend this much)
  # school zone plate (pentagon above the MAXIMUM sign): shown with the limit it is mounted on. Ontario posts no end of
  # school zone sign, so the plate goes when another limit is accepted, the road changes (the limit is dropped), or after
  # school_hold_distance without another school sign.
  school_min_prob: float = 0.6  # per-reading school probability
  school_min_confidence: float = 0.3  # the reading's value confidence (min_confidence for the limit itself)
  school_min_hits: int = 2  # school readings of the held limit's value within school_window
  school_window: float = 4.  # s
  school_hold_distance: float = 500.  # m since the last school reading


@dataclass
class SignFilterState:
  speed_limit: int = 0  # in sign units, 0 when unknown
  confidence: float = 0.
  time_since_accept: float = 0.
  distance_since_accept: float = 0.
  school_zone: bool = False  # the held limit is a school zone limit
  distance_since_school: float = 0.  # m since the last school reading (while school_zone)


@dataclass
class SignFilter:
  params: FilterParams = field(default_factory=FilterParams)
  is_metric: bool = True  # sign units; signd follows the IsMetric param

  def __post_init__(self):
    self._hits: deque[tuple[float, Detection]] = deque()
    self._school_hits: deque[tuple[float, int]] = deque()  # (t, value) of readings with a school pentagon
    self._yaw: deque[tuple[float, float]] = deque()  # (t, heading change during step)
    self._last_t: float | None = None
    self._current_road = ''  # latest non-empty road name
    self._road = ''  # road the held limit belongs to, '' if unknown
    self._new_road = ''  # different road name being driven on
    self._new_road_dist = 0.  # m driven on it
    self._off_road_dist = 0.  # m driven with no road name since the held road
    self._map_limit = 0  # map speed limit in sign units, 0 if unknown
    self.state = SignFilterState()

  def reset(self) -> None:
    self._hits.clear()
    self._school_hits.clear()
    self._yaw.clear()
    self._road = self._new_road = ''
    self._new_road_dist = self._off_road_dist = 0.
    self.state = SignFilterState()

  def update(self, t: float, detections: list[Detection], v_ego: float | None = None, yaw_rate: float = 0.,
             road: str = '', map_limit: float = 0.) -> SignFilterState:
    """v_ego None: unknown (e.g. offline without logs), distance resets are off.
    road: road name from the map, '' if not known right now. map_limit: the map's limit in sign units, 0 if unknown."""
    self._current_road = road or self._current_road
    self._map_limit = round(map_limit)
    dt = 0. if self._last_t is None else max(0., t - self._last_t)
    self._last_t = t

    self._update_hold(t, dt, v_ego or 0., yaw_rate, road)

    p = self.params
    # no standstill rule: signs are often read while stopped beside them at an intersection. What else is in view at a
    # standstill (a licence plate ahead at a red light) passes the growth check too, so the classifier must reject it.
    new_school = False
    new_hit = False
    for det in detections:
      if not plausible(det.value, self.is_metric, road or self._current_road):
        continue
      if det.confidence >= p.min_confidence:
        self._hits.append((t, det))
        new_hit = True
      # a school reading only needs its value to match the limit accepted from the confident readings: it can't change
      # the limit, so a less confident read of the same sign (often the far, first sightings) still counts
      if det.value and det.school >= p.school_min_prob and det.confidence >= p.school_min_confidence:
        self._school_hits.append((t, det.value))
        new_school = True
    while self._hits and t - self._hits[0][0] > p.window:
      self._hits.popleft()
    while self._school_hits and t - self._school_hits[0][0] > p.school_window:
      self._school_hits.popleft()

    # only on a new reading: re-accepting from readings already counted (e.g. when the map limit changes and one old
    # reading suddenly agrees with it) would restart the turn and road tracking for a sign that may be behind the car
    candidate = self._candidate() if new_hit else None
    if candidate is not None:
      value, confidence = candidate
      self._hits.clear()
      self._yaw.clear()
      self._road, self._new_road, self._new_road_dist = self._current_road, '', 0.
      self._off_road_dist = 0.
      # the same limit read again (often the same sign, or a repeater) keeps its school plate; another limit drops it
      same = value == self.state.speed_limit
      self.state = SignFilterState(speed_limit=value, confidence=confidence,
                                   school_zone=same and self.state.school_zone,
                                   distance_since_school=self.state.distance_since_school if same else 0.)

    # checked on a new school reading or a newly accepted limit, so the hold distance counts from the last school reading
    limit = self.state.speed_limit
    if (new_school or candidate is not None) and limit and sum(v == limit for _, v in self._school_hits) >= p.school_min_hits:
      self.state.school_zone = True
      self.state.distance_since_school = 0.
    return self.state

  def _update_hold(self, t: float, dt: float, v_ego: float, yaw_rate: float, road: str) -> None:
    if self.state.speed_limit == 0:
      return

    p = self.params
    self.state.time_since_accept += dt
    step = max(0., v_ego) * dt
    self.state.distance_since_accept += step
    if self.state.school_zone:
      self.state.distance_since_school += step
      if self.state.distance_since_school > p.school_hold_distance:
        self.state.school_zone = False

    self._yaw.append((t, yaw_rate * dt))
    while self._yaw and t - self._yaw[0][0] > p.turn_window:
      self._yaw.popleft()
    heading_change = abs(sum(d for _, d in self._yaw))

    turned = heading_change > p.turn_reset_angle
    dist = self.state.distance_since_accept

    if not self._road:
      self._road = road  # the limit was accepted before the map knew the road
      if not self._road:
        if self.state.time_since_accept > p.max_hold_time or dist > p.max_hold_distance or turned:
          self.reset()
        return

    if not road:
      self._off_road_dist += step
      if turned and self._off_road_dist >= p.off_road_distance:
        self.reset()
        return
    else:
      self._off_road_dist = 0.

    if road == self._road:
      self._new_road, self._new_road_dist = '', 0.
    elif road:
      if road != self._new_road:
        self._new_road, self._new_road_dist = road, 0.
      self._new_road_dist += step
      if turned or self._new_road_dist >= p.road_change_distance:
        if dist > p.road_grace_distance or turned:
          self.reset()
          return
        self._road, self._new_road, self._new_road_dist = road, '', 0.  # read just before the name caught up

    if dist > p.max_hold_distance_on_road:
      self.reset()

  def _candidate(self) -> tuple[int, float] | None:
    by_value: dict[int, list[Detection]] = {}
    for _, det in self._hits:
      by_value.setdefault(det.value, []).append(det)

    best: tuple[int, float] | None = None
    for value, dets in by_value.items():
      if value and self._map_limit:
        need = self.params.min_hits_map_agree if value == self._map_limit else self.params.min_hits_map_disagree
      else:
        need = self.params.min_hits
      if len(dets) < need:
        continue
      if growth(dets[0], dets[-1]) < self.params.min_area_growth:
        continue
      confidence = sum(d.confidence for d in dets) / len(dets)
      if best is None or confidence > best[1]:
        best = (value, confidence)
    return best
