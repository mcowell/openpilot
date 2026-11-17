"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Close lead stop: request a stop when crawling up close behind a stopped lead.

openpilot only stops (shouldStop) below 0.3 m/s. Above that, approaching a stopped car, the planner asks for a very
gentle deceleration (-0.1 to -0.3 m/s^2) that some cars do not realize at walking pace: they hold their idle creep at
1-2 km/h and close the last metres to the lead (seen on a VW MQB with a vision-only lead, 2-4 m gaps before the driver
braked). Within this window the planner already wants to slow down, so this only asks for the stop earlier; it never
adds braking anywhere else, and lets go as soon as the lead moves off.
"""
from dataclasses import dataclass

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD


@dataclass(frozen=True)
class CloseLeadStopParams:
  v_ego_max: float = 0.85  # m/s (3 km/h): crawling (creep seen at 1.2-3 km/h)
  d_rel_on: float = 5.0  # m, lead this close to start the stop
  d_rel_off: float = 6.5  # m, lead this far to let go (hysteresis)
  v_lead_on: float = 0.5  # m/s, lead essentially stopped
  v_lead_off: float = 1.0  # m/s, lead moving off: let go
  a_target_max: float = 0.0  # m/s^2, only when the planner already wants to slow down


DEFAULT_PARAMS = CloseLeadStopParams()


def close_lead_stop(active: bool, v_ego: float, lead_present: bool, d_rel: float, v_lead: float, a_target: float,
                    p: CloseLeadStopParams = DEFAULT_PARAMS) -> bool:
  """Next state of the close lead stop request (active: its state on the previous step)"""
  if not lead_present:
    return False
  if active:
    return d_rel < p.d_rel_off and v_lead < p.v_lead_off
  return v_ego < p.v_ego_max and d_rel < p.d_rel_on and v_lead < p.v_lead_on and a_target < p.a_target_max


class CloseLeadStop:
  def __init__(self, params: CloseLeadStopParams | None = None):
    self.p = params or DEFAULT_PARAMS
    self.params = Params()
    self.enabled = self.params.get_bool("CloseLeadStop")
    self.frame = -1
    self.active = False

  def update(self, enabled: bool, v_ego: float, lead, a_target: float) -> bool:
    """lead: radarState.leadOne. Returns whether to request a stop."""
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("CloseLeadStop")
    if not (self.enabled and enabled):
      self.active = False
      return False
    self.active = close_lead_stop(self.active, v_ego, lead.present, lead.dRel, lead.vLead, a_target, self.p)
    return self.active
