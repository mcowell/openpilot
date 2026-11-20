"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp

DESCRIPTIONS = {
  'cluster_speed_limit': tr_noop(
    'Show the speed limit sunnypilot is using as the traffic sign in the instrument cluster. ' +
    'Requires traffic sign recognition (VZE) coded in the camera and instrument cluster. Applies after the car is restarted.'
  ),
  'cluster_no_passing': tr_noop(
    'Also show the camera\'s own no-passing sign (from its map data) next to the speed limit on the cluster\'s ' +
    'Traffic signs page. Applies after the car is restarted.'
  ),
}


class VolkswagenSettings(BrandSettings):
  def __init__(self):
    super().__init__()

    self.cluster_speed_limit = toggle_item_sp(
      lambda: tr("Speed Limit in Instrument Cluster (Experimental)"),
      description=lambda: tr(DESCRIPTIONS["cluster_speed_limit"]),
      initial_state=ui_state.params.get_bool("VolkswagenClusterSpeedLimit"),
      callback=lambda state: ui_state.params.put_bool("VolkswagenClusterSpeedLimit", state),
      enabled=lambda: ui_state.is_offroad(),
    )

    self.cluster_no_passing = toggle_item_sp(
      lambda: tr("No-Passing Sign in Instrument Cluster (Experimental)"),
      description=lambda: tr(DESCRIPTIONS["cluster_no_passing"]),
      initial_state=ui_state.params.get_bool("VolkswagenClusterNoPassing"),
      callback=lambda state: ui_state.params.put_bool("VolkswagenClusterNoPassing", state),
      enabled=lambda: ui_state.is_offroad() and ui_state.params.get_bool("VolkswagenClusterSpeedLimit"),
    )

    self.items = [
      self.cluster_speed_limit,
      self.cluster_no_passing,
    ]

  def update_settings(self):
    pass
