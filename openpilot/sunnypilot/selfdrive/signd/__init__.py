"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
SIGND_RATE_HZ = 2.  # Hz, liveVisionSpeedLimitSP publish rate. Detector work is scheduled by reader.SignReader

# Values printed on regulatory maximum speed signs (same set for km/h and mph, the classifier picks one)
SPEED_LIMIT_VALUES = tuple(range(10, 135, 5))
