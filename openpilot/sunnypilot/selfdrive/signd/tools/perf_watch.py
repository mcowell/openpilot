#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Live check that signd isn't hurting the driving model. Run on the device (onroad, e.g. parked with ignition on)
and toggle the SpeedLimitVision setting to compare:

  python -m openpilot.sunnypilot.selfdrive.signd.tools.perf_watch [--seconds 600]

Prints one line per second, then a summary of signd on vs off.
"""
import argparse
import statistics
import time

from openpilot.cereal import messaging


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seconds', type=float, default=600.)
  args = ap.parse_args()

  sm = messaging.SubMaster(['modelV2', 'deviceState', 'liveVisionSpeedLimitSP', 'longitudinalPlanSP'])
  rows: list[dict] = []
  signd_frames: list[float] = []
  start = last_print = time.monotonic()
  print('   t  signd  Hz  exec_ms  limit | modeld drop%  exec_ms | cpu% (max core)  gpu% | temp C: gpu  cpu  max  fan% thermal | source')
  while time.monotonic() - start < args.seconds:
    sm.update(100)
    now = time.monotonic()
    if sm.updated['liveVisionSpeedLimitSP']:
      signd_frames.append(now)
    if now - last_print < 1.:
      continue
    last_print = now

    signd_frames = [t for t in signd_frames if now - t < 5.]
    signd_on = sm.alive['liveVisionSpeedLimitSP'] and len(signd_frames) > 0
    vision, model, dev = sm['liveVisionSpeedLimitSP'], sm['modelV2'], sm['deviceState']
    cpu = list(dev.cpuUsagePercent)
    row = {
      'signd': signd_on,
      'signd_hz': len(signd_frames) / 5.,
      'signd_ms': vision.modelExecutionTime * 1000 if signd_on else float('nan'),
      'drop': model.frameDropPerc,
      'model_ms': model.modelExecutionTime * 1000,
      'cpu': statistics.mean(cpu) if cpu else 0.,
      'cpu_max': max(cpu) if cpu else 0,
      'gpu': dev.gpuUsagePercent,
      'temp': dev.maxTempC,
      'gpu_temp': max(dev.gpuTempC, default=0.),
      'cpu_temp': max(dev.cpuTempC, default=0.),
      'fan': dev.fanSpeedPercentDesired,
    }
    rows.append(row)
    limit = f'{vision.speedLimitSign:3d}' if signd_on and vision.speedLimitValid else '  -'
    print(' '.join([
      f"{now - start:4.0f}  {'on ' if signd_on else 'off'}  {row['signd_hz']:3.1f}  {row['signd_ms']:6.1f}   {limit}  |",
      f"      {row['drop']:5.1f}  {row['model_ms']:7.1f} |",
      f"{row['cpu']:4.0f} ({row['cpu_max']:3d})      {row['gpu']:3d} |",
      f"        {row['gpu_temp']:4.0f} {row['cpu_temp']:4.0f} {row['temp']:4.0f}  {row['fan']:3d}  {str(dev.thermalStatus):7s} |",
      str(sm['longitudinalPlanSP'].speedLimit.resolver.source),
    ]), flush=True)

  summarize(rows)


def summarize(rows: list[dict]) -> None:
  for state in (True, False):
    sel = [r for r in rows if r['signd'] == state]
    if not sel:
      continue

    def med(k: str, sel=sel) -> float:
      return statistics.median(r[k] for r in sel)

    parts = [f"\nsignd {'on ' if state else 'off'} ({len(sel)} s):",
             f"modeld drop {max(r['drop'] for r in sel):.1f}% max, exec {med('model_ms'):.1f} ms;",
             f"cpu {med('cpu'):.0f}%, gpu {med('gpu'):.0f}%;",
             f"gpu temp {sel[0]['gpu_temp']:.0f} -> {sel[-1]['gpu_temp']:.0f} C, max temp {max(r['temp'] for r in sel):.0f} C, fan {med('fan'):.0f}%"]
    if state:
      parts.append(f"; signd {med('signd_hz'):.1f} Hz, {med('signd_ms'):.0f} ms median, {max(r['signd_ms'] for r in sel):.0f} ms max")
    print(' '.join(parts))


if __name__ == '__main__':
  main()
