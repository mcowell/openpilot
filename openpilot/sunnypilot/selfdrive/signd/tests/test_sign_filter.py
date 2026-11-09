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


def run(f, seq, v_ego=25., yaw_rate=0., road=''):
  state = None
  for t, dets in seq:
    state = f.update(t, dets, v_ego, yaw_rate, road)
  return state


def drive(f, seconds, start_t, v_ego=30., yaw_rate=0., road=''):
  """No signs in view for this long at 20 Hz; returns the time at the end"""
  t = start_t
  for _ in range(round(seconds * 20)):
    t += 0.05
    f.update(t, [], v_ego, yaw_rate, road)
  return t


class TestSignFilter(OpenpilotTestCase):
  def test_accepts_after_min_hits(self):
    f = SignFilter()
    n = f.params.min_hits
    seq = approaching(80, n)
    assert run(f, seq[:n - 1]).speed_limit == 0
    assert run(f, seq[n - 1:]).speed_limit == 80

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
    seq = [(i * 1.1, dets) for i, (_, dets) in enumerate(approaching(50, 3))]  # hits 1.1 s apart, never 2 within 1 s
    assert run(f, seq).speed_limit == 0

  def test_receding_sign_rejected(self):
    f = SignFilter()
    assert run(f, approaching(100, 5, start_size=0.1, growth=0.8)).speed_limit == 0

  def test_sign_leaving_over_the_top_accepted(self):
    # Southwood Dr, 2026-11-07: read twice as it left the top of the frame. Cut off by the edge, its box got shorter
    # (area shrank) while it kept getting wider: still being approached
    f = SignFilter()
    seq = [(0., [Detection(40, 0.75, (0.652, 0.007, 0.041, 0.093))]), (0.15, [Detection(40, 0.74, (0.660, 0.0, 0.045, 0.080))])]
    assert run(f, seq).speed_limit == 40

  def test_sign_leaving_over_the_top_and_shrinking_rejected(self):
    f = SignFilter()
    seq = [(0., [Detection(40, 0.9, (0.65, 0.0, 0.045, 0.08))]), (0.15, [Detection(40, 0.9, (0.66, 0.0, 0.040, 0.06))])]
    assert run(f, seq).speed_limit == 0

  def test_sign_read_while_stopped_accepted(self):
    # stopped at an intersection beside the sign: it doesn't grow, but it's the limit (a stationary licence plate looks the
    # same to the filter, so rejecting plates is the classifier's job)
    f = SignFilter()
    seq = [(i * DT, [Detection(80, 0.9, (0.75, 0.2, 0.036, 0.066))]) for i in range(3)]
    assert run(f, seq, v_ego=0.).speed_limit == 80

  def test_end_of_zone_clears_limit(self):
    f = SignFilter()
    run(f, approaching(40, 3))
    # "40 AREA ENDS" sign: the classifier reports end of zone as value 0
    state = run(f, approaching(0, 3, start_t=30.))
    assert state.speed_limit == 0

  def test_single_end_misread_keeps_limit(self):
    f = SignFilter()
    run(f, approaching(40, 3))
    assert run(f, [(30., [Detection(0, 0.9, (0.7, 0.4, 0.05, 0.06))])]).speed_limit == 40

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

  def test_road_hold_outlasts_old_caps(self):
    # Highway 416, 2026-11-08: signs 8+ km apart, limit dropped every 5 km
    f = SignFilter()
    assert run(f, approaching(110, 3), road='Highway 416').speed_limit == 110
    t = drive(f, 500., 1., road='Highway 416')  # 15 km, over 8 minutes
    assert f.state.speed_limit == 110
    drive(f, 200., t, road='Highway 416')  # 21 km: capped
    assert f.state.speed_limit == 0

  def test_time_limit_only_without_road_name(self):
    f = SignFilter()
    run(f, approaching(60, 3), road='Merivale Road')
    drive(f, 900., 1., v_ego=0.1, road='Merivale Road')  # a long jam
    assert f.state.speed_limit == 60

  def test_road_change_clears_limit(self):
    # Robertson Road -> Baseline Road, 2026-11-08: straight on, so no turn; 80 was held for 216 s into a 70 zone
    f = SignFilter()
    run(f, approaching(80, 3), road='Robertson Road')
    t = drive(f, 15., 1., road='Robertson Road')
    assert f.state.speed_limit == 80
    t = drive(f, 2., t, road='Baseline Road')  # 60 m on it: could be map matching at an intersection
    assert f.state.speed_limit == 80
    drive(f, 2., t, road='Baseline Road')
    assert f.state.speed_limit == 0

  def test_brief_other_road_name_keeps_limit(self):
    f = SignFilter()
    run(f, approaching(60, 3), road='Old Richmond Road')
    t = drive(f, 15., 1., road='Old Richmond Road')
    t = drive(f, 2., t, road='West Hunt Club Road')
    t = drive(f, 10., t, road='Old Richmond Road')
    drive(f, 2., t, road='West Hunt Club Road')
    assert f.state.speed_limit == 60

  def test_empty_road_name_keeps_limit(self):
    f = SignFilter()
    run(f, approaching(80, 3), road='Robertson Road')
    drive(f, 60., 1., road='')
    assert f.state.speed_limit == 80

  def test_stopped_on_cross_street_name_keeps_limit(self):
    f = SignFilter()
    run(f, approaching(60, 3), road='Merivale Road')
    t = drive(f, 15., 1., road='Merivale Road')
    drive(f, 60., t, v_ego=0., road='Clyde Avenue')  # no distance on it
    assert f.state.speed_limit == 60

  def test_sign_just_before_name_change_belongs_to_new_road(self):
    f = SignFilter()
    run(f, approaching(40, 3), road='Baseline Road')
    t = drive(f, 2., 1., v_ego=10., road='Baseline Road')
    drive(f, 15., t, v_ego=10., road='Southwood Drive')
    assert f.state.speed_limit == 40

  def test_bend_on_same_road_keeps_limit(self):
    # a corner on Greenview Avenue, 2026-11-08
    f = SignFilter()
    run(f, approaching(40, 3), road='Greenview Avenue')
    drive(f, 5., 1., v_ego=8., yaw_rate=math.radians(25.), road='Greenview Avenue')
    assert f.state.speed_limit == 40

  def test_turn_onto_new_road_clears_at_once(self):
    f = SignFilter()
    run(f, approaching(60, 3), road='Baseline Road')
    t = drive(f, 15., 1., road='Baseline Road')
    t = drive(f, 4., t, v_ego=6., yaw_rate=math.radians(25.), road='Baseline Road')  # 100 degrees, name not updated yet
    assert f.state.speed_limit == 60
    drive(f, 0.2, t, v_ego=6., yaw_rate=math.radians(25.), road='Southwood Drive')
    assert f.state.speed_limit == 0

  def test_lane_change_does_not_reset(self):
    f = SignFilter()
    run(f, approaching(110, 3))
    left = [(1. + i * DT, []) for i in range(10)]
    right = [(3. + i * DT, []) for i in range(10)]
    run(f, left, yaw_rate=math.radians(5.))
    state = run(f, right, yaw_rate=-math.radians(5.))
    assert state.speed_limit == 110
