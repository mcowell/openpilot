"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
SIGND_RATE_HZ = 2.  # Hz, sign detection rate; signs stay in view for seconds, and the 3X GPU is shared with the driving model (~80 ms per run)

# Values printed on regulatory maximum speed signs (same set for km/h and mph, the classifier picks one)
SPEED_LIMIT_VALUES = tuple(range(10, 135, 5))
