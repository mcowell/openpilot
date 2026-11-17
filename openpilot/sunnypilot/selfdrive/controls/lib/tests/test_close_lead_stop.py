"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.selfdrive.controls.lib.close_lead_stop import close_lead_stop

CRAWL = 0.35  # m/s, the 1.2-1.5 km/h creep seen before the driver braked


def test_crawling_up_to_stopped_lead_stops():
  assert close_lead_stop(False, CRAWL, True, 4.0, 0.1, -0.15)


def test_not_while_still_far():
  assert not close_lead_stop(False, CRAWL, True, 7.0, 0.1, -0.15)


def test_not_above_crawl_speed():
  assert not close_lead_stop(False, 1.5, True, 4.0, 0.1, -0.5)


def test_not_when_lead_moving():
  assert not close_lead_stop(False, CRAWL, True, 4.0, 1.2, -0.15)


def test_not_when_planner_accelerates():
  assert not close_lead_stop(False, CRAWL, True, 4.0, 0.1, 0.2)


def test_not_without_lead():
  assert not close_lead_stop(False, CRAWL, False, 0.0, 0.0, -0.15)
  assert not close_lead_stop(True, 0.0, False, 0.0, 0.0, -0.15)


def test_holds_while_stopped_behind_lead():
  # stopped: speed 0, planner's request may be about 0; stays a stop while the lead stays put
  assert close_lead_stop(True, 0.0, True, 4.0, 0.0, 0.0)
  assert close_lead_stop(True, 0.0, True, 6.0, 0.3, 0.3)  # inside the hysteresis band


def test_lets_go_when_lead_moves_off():
  assert not close_lead_stop(True, 0.0, True, 7.0, 0.5, 0.3)
  assert not close_lead_stop(True, 0.0, True, 5.0, 1.5, 0.5)
