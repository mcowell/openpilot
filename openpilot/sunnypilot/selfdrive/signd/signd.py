#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

signd: reads speed limit signs from the road camera and publishes liveVisionSpeedLimitSP.
Runs only when SpeedLimitVision is enabled (see process_config.py).
"""
import os
import time
from dataclasses import dataclass

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
from openpilot.sunnypilot.selfdrive.signd.guard import ModelGuard
from openpilot.sunnypilot.selfdrive.signd.reader import SignReader
from openpilot.sunnypilot.selfdrive.signd.sign_filter import SignFilter, SignFilterState

CAMERA = 'road'


@dataclass
class DebugStatus:
  guard_level: int = 0
  tracking: bool = False
  last_read_sign: int = 0
  time_since_read: float = -1.


def get_vision_speed_limit_packet(state: SignFilterState, is_metric: bool, frame_id: int, exec_time: float,
                                  debug: DebugStatus | None = None) -> messaging.capnp._DynamicStructBuilder:
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
  if debug is not None:
    vsl.guardLevel = debug.guard_level
    vsl.tracking = debug.tracking
    vsl.lastReadSign = debug.last_read_sign
    vsl.timeSinceRead = debug.time_since_read
  return msg


def wait_for_model(sm: messaging.SubMaster, frame_id: int, timeout: float = 0.045) -> None:
  """Until modeld has published its output for frame_id (or timeout, about a frame)"""
  deadline = time.monotonic() + timeout
  while sm['modelV2'].frameId < frame_id and time.monotonic() < deadline:
    sm.update(5)


def main():
  # lowest GPU priority (KGSL: 0 highest, 15 lowest), below the models' 12 from launch_env.sh, so modeld's work
  # preempts ours instead of waiting behind an ~80 ms detector run. Must be set before tinygrad opens the GPU.
  os.environ['QCOM_PRIORITY'] = '15'
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
  reader = SignReader(detector)
  guard = ModelGuard()  # backs off if the driving model drops frames
  sign_filter = SignFilter()
  cloudlog.warning("signd: models loaded, starting")

  sm = messaging.SubMaster(['carState', 'deviceMotion', 'modelV2'])
  pm = messaging.PubMaster(['liveVisionSpeedLimitSP'])

  is_metric = params.get_bool("IsMetric")
  last_params_update = last_publish = exec_time = 0.
  last_read: tuple[int, float] | None = None  # (value, time) of the latest single reading
  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    now = time.monotonic()
    sm.update(0)
    if now - last_params_update > PARAMS_UPDATE_PERIOD:
      last_params_update = now
      new_is_metric = params.get_bool("IsMetric")
      if new_is_metric != is_metric:
        sign_filter.reset()  # held value is in the old units
      is_metric = new_is_metric

    prev_level = guard.level
    level = guard.update(now, sm['modelV2'].frameDropPerc)
    if level != prev_level:
      cloudlog.warning(f"signd: guard {prev_level.name} -> {level.name} (modelV2.frameDropPerc {sm['modelV2'].frameDropPerc:.2f})")

    # every frame goes to the reader, which picks this frame's GPU job, if any, and does its CPU work now, while modeld is
    # busy with the same frame. The job runs once modeld has finished it: the GPU is then free until the next frame.
    detections = None
    job = reader.prepare(now, np.frombuffer(buf.data, dtype=np.uint8), buf.stride, buf.uv_offset, buf.width, buf.height, level)
    if job is not None:
      wait_for_model(sm, vipc_client.frame_id)
      t1 = time.perf_counter()
      detections = reader.run(job)
      exec_time = time.perf_counter() - t1
      if detections:
        last_read = (max(detections, key=lambda d: d.confidence).value, now)

    yaw_rate = -sm['deviceMotion'].angularVelocityDevice.z  # device z points down
    state = sign_filter.update(now, detections or [], sm['carState'].vEgo, yaw_rate)
    if now - last_publish >= 1. / SIGND_RATE_HZ:
      last_publish = now
      debug = DebugStatus(int(level), bool(reader.tracks), last_read[0] if last_read else 0, now - last_read[1] if last_read else -1.)
      pm.send('liveVisionSpeedLimitSP', get_vision_speed_limit_packet(state, is_metric, vipc_client.frame_id, exec_time, debug))


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
