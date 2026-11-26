"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.selfdrive.signd.work_zone import WorkZoneFilter, WorkZoneParams


def drive(f, t0, t1, v=15., reads=None):
  """update at 20 Hz from t0 to t1; reads: {t: probability} handed over at that update"""
  reads = reads or {}
  t = t0
  while t < t1 - 1e-9:
    s = f.update(t, [(t, reads[round(t, 2)])] if round(t, 2) in reads else [], v)
    t = round(t + 0.05, 2)
  return s


class TestWorkZone:
  def test_two_readings_start_the_zone(self):
    f = WorkZoneFilter()
    assert not drive(f, 0., 1., reads={0.5: 0.9}).active  # one reading isn't enough
    assert drive(f, 1., 2., reads={1.5: 0.8}).active  # a second within 3 s

  def test_readings_too_far_apart_or_too_weak(self):
    f = WorkZoneFilter()
    assert not drive(f, 0., 6., reads={0.5: 0.9, 4.: 0.9}).active
    f = WorkZoneFilter()
    assert not drive(f, 0., 2., reads={0.5: 0.6, 1.: 0.65, 1.5: 0.69}).active

  def test_ends_after_hold_distance_from_the_last_sign(self):
    f = WorkZoneFilter(WorkZoneParams(hold_distance=1000.))
    assert drive(f, 0., 1., reads={0.2: 0.9, 0.4: 0.9}).active
    assert drive(f, 1., 60., v=15.).active  # ~890 m
    # another road work sign restarts the distance
    assert drive(f, 60., 61., reads={60.2: 0.9, 60.4: 0.9}).active
    assert drive(f, 61., 120., v=15.).active
    assert not drive(f, 120., 130., v=15.).active  # > 1000 m since the last sign

  def test_stopped_in_the_zone_keeps_it(self):
    f = WorkZoneFilter(WorkZoneParams(hold_distance=100.))
    drive(f, 0., 1., reads={0.2: 0.9, 0.4: 0.9})
    assert drive(f, 1., 600., v=0.).active
