#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

signd: reads speed limit signs from the road camera and publishes liveVisionSpeedLimitSP.
Runs only when SpeedLimitVision is enabled (see process_config.py).
"""
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.signd import SIGND_RATE_HZ
from openpilot.sunnypilot.selfdrive.signd.detector import SignDetector
from openpilot.sunnypilot.selfdrive.signd.sign_filter import SignFilter, SignFilterState

CAMERA = 'road'


def get_vision_speed_limit_packet(state: SignFilterState, is_metric: bool, frame_id: int, exec_time: float) -> messaging.capnp._DynamicStructBuilder:
  msg = messaging.new_message('liveVisionSpeedLimitSP', valid=True)
  vsl = msg.liveVisionSpeedLimitSP
  vsl.speedLimitValid = state.speed_limit > 0
  vsl.speedLimit = float(state.speed_limit * (CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS))
  vsl.speedLimitSign = state.speed_limit
  vsl.isMetric = is_metric
  vsl.confidence = state.confidence
  vsl.timeSinceSign = state.time_since_accept
  vsl.distanceSinceSign = state.distance_since_accept
  vsl.frameId = frame_id
  vsl.modelExecutionTime = exec_time
  return msg


def main():
  # not realtime: keep off modeld/dmonitoringmodeld's core and don't preempt anything, a sign read can wait
  set_core_affinity([0, 1, 2, 3])
  params = Params()

  if not SignDetector.available():
    cloudlog.error("signd: no sign models found, exiting")
    return

  cloudlog.warning("signd: connecting to road stream")
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True)
  while not vipc_client.connect(False):
    time.sleep(0.1)

  detector = SignDetector(CAMERA)
  sign_filter = SignFilter()
  cloudlog.warning("signd: models loaded, starting")

  sm = messaging.SubMaster(['carState', 'deviceMotion'])
  pm = messaging.PubMaster(['liveVisionSpeedLimitSP'])

  is_metric = params.get_bool("IsMetric")
  last_params_update = last_run = 0.
  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    now = time.monotonic()
    if now - last_run < 1. / SIGND_RATE_HZ:
      continue
    last_run = now

    sm.update(0)
    if now - last_params_update > PARAMS_UPDATE_PERIOD:
      last_params_update = now
      new_is_metric = params.get_bool("IsMetric")
      if new_is_metric != is_metric:
        sign_filter.reset()  # held value is in the old units
      is_metric = new_is_metric

    t1 = time.perf_counter()
    detections = detector.detect_nv12(np.frombuffer(buf.data, dtype=np.uint8), buf.stride, buf.uv_offset, buf.width, buf.height)
    t2 = time.perf_counter()

    yaw_rate = -sm['deviceMotion'].angularVelocityDevice.z  # device z points down
    state = sign_filter.update(now, detections, sm['carState'].vEgo, yaw_rate)
    pm.send('liveVisionSpeedLimitSP', get_vision_speed_limit_packet(state, is_metric, vipc_client.frame_id, t2 - t1))


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
