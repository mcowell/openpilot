"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.signd.sign_filter import Detection, FilterParams, SignFilter

DT = 0.2


def approaching(value, n, start_t=0., confidence=0.9, start_size=0.02, growth=1.15):
  """n detections of one sign growing in the frame, as when driving towards it"""
  out = []
  size = start_size
  for i in range(n):
    out.append((start_t + i * DT, [Detection(value, confidence, (0.7, 0.4, size, size * 1.2))]))
    size *= growth
  return out


def run(f, seq, v_ego=25., yaw_rate=0.):
  state = None
  for t, dets in seq:
    state = f.update(t, dets, v_ego, yaw_rate)
  return state


class TestSignFilter(OpenpilotTestCase):
  def test_accepts_after_min_hits(self):
    f = SignFilter()
    seq = approaching(80, 3)
    assert run(f, seq[:2]).speed_limit == 0
    assert run(f, seq[2:]).speed_limit == 80

  def test_low_confidence_ignored(self):
    f = SignFilter()
    assert run(f, approaching(80, 10, confidence=0.3)).speed_limit == 0

  def test_single_misread_does_not_win(self):
    f = SignFilter()
    seq = approaching(60, 5)
    seq[2] = (seq[2][0], [Detection(80, 0.9, seq[2][1][0].box)])
    assert run(f, seq).speed_limit == 60

  def test_hits_must_be_within_window(self):
    f = SignFilter(FilterParams(window=1.0))
    seq = [(i * 0.9, dets) for i, (_, dets) in enumerate(approaching(50, 3))]  # hits 0.9 s apart, never 3 within 1 s
    assert run(f, seq).speed_limit == 0

  def test_receding_sign_rejected(self):
    f = SignFilter()
    assert run(f, approaching(100, 5, start_size=0.1, growth=0.8)).speed_limit == 0

  def test_new_sign_replaces_old(self):
    f = SignFilter()
    run(f, approaching(100, 3))
    state = run(f, approaching(70, 3, start_t=30.))
    assert state.speed_limit == 70

  def test_holds_between_signs(self):
    f = SignFilter()
    run(f, approaching(90, 3))
    state = run(f, [(1. + i * DT, []) for i in range(100)], v_ego=25.)
    assert state.speed_limit == 90

  def test_reset_after_distance(self):
    f = SignFilter(FilterParams(max_hold_distance=1000.))
    run(f, approaching(90, 3))
    state = run(f, [(1. + i, []) for i in range(60)], v_ego=25.)  # 1500 m
    assert state.speed_limit == 0

  def test_reset_after_time(self):
    f = SignFilter(FilterParams(max_hold_time=60.))
    run(f, approaching(40, 3))
    state = run(f, [(1. + i, []) for i in range(70)], v_ego=0.)
    assert state.speed_limit == 0

  def test_reset_after_turn(self):
    f = SignFilter()
    run(f, approaching(50, 3))
    yaw_rate = math.radians(90.) / 4.  # 90 deg turn over 4 s
    state = run(f, [(1. + i * DT, []) for i in range(25)], v_ego=8., yaw_rate=yaw_rate)
    assert state.speed_limit == 0

  def test_lane_change_does_not_reset(self):
    f = SignFilter()
    run(f, approaching(110, 3))
    left = [(1. + i * DT, []) for i in range(10)]
    right = [(3. + i * DT, []) for i in range(10)]
    run(f, left, yaw_rate=math.radians(5.))
    state = run(f, right, yaw_rate=-math.radians(5.))
    assert state.speed_limit == 110
