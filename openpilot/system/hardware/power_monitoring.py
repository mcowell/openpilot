import time
import threading

from openpilot.common.params import Params
from openpilot.common.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.system.statsd import statlog

CAR_VOLTAGE_LOW_PASS_K = 0.011 # LPF gain for 45s tau (dt/tau / (dt/tau + 1))

# While driving, a battery charges completely in about 30-60 minutes
CAR_BATTERY_CAPACITY_uWh = 30e6
CAR_CHARGING_RATE_W = 45

VBATT_PAUSE_CHARGING = 11.8           # Lower limit on the LPF car battery voltage

# With the ignition off, only a battery charger or maintainer holds the car battery this high for long (a rested battery
# sits at ~12.6-13 V, surface charge after a drive fades within minutes). While it does, the battery is charging, not
# draining: refill the capacity budget instead of spending it, so a car on a charger doesn't shut the device down.
VBATT_EXTERNAL_CHARGING = 13.3        # LPF car battery voltage, V
EXTERNAL_CHARGING_MIN_TIME_S = 600    # sustained, after the ignition was last on
EXTERNAL_CHARGING_RATE_W = 10         # conservative: maintainers supply ~10-50 W
MAX_TIME_OFFROAD_S = 30*3600
MIN_ON_TIME_S = 3600
DELAY_SHUTDOWN_TIME_S = 300 # Wait at least DELAY_SHUTDOWN_TIME_S seconds after offroad_time to shutdown.
VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S = 60

class PowerMonitoring:
  def __init__(self):
    self.params = Params()
    self.last_measurement_time = None           # Used for integration delta
    self.last_save_time = 0                     # Used for saving current value in a param
    self.power_used_uWh = 0                     # Integrated power usage in uWh since going into offroad
    self.next_pulsed_measurement_time = None
    self.car_voltage_mV = 12e3                  # Low-passed version of peripheralState voltage
    self.car_voltage_instant_mV = 12e3          # Last value of peripheralState voltage
    self.integration_lock = threading.Lock()
    self.external_charging_since: float | None = None  # when the LPF voltage last rose above VBATT_EXTERNAL_CHARGING

    car_battery_capacity_uWh = self.params.get("CarBatteryCapacity") or 0

    # Reset capacity if it's low
    self.car_battery_capacity_uWh = max((CAR_BATTERY_CAPACITY_uWh / 10), car_battery_capacity_uWh)

  # Calculation tick
  def calculate(self, voltage: float | None, ignition: bool):
    try:
      now = time.monotonic()

      # If peripheralState is None, we're probably not in a car, so we don't care
      if voltage is None:
        with self.integration_lock:
          self.last_measurement_time = None
          self.next_pulsed_measurement_time = None
          self.power_used_uWh = 0
        return

      # Low-pass battery voltage
      self.car_voltage_instant_mV = voltage
      self.car_voltage_mV = ((voltage * CAR_VOLTAGE_LOW_PASS_K) + (self.car_voltage_mV * (1 - CAR_VOLTAGE_LOW_PASS_K)))
      statlog.gauge("car_voltage", self.car_voltage_mV / 1e3)

      # Cap the car battery power and save it in a param every 10-ish seconds
      self.car_battery_capacity_uWh = max(self.car_battery_capacity_uWh, 0)
      self.car_battery_capacity_uWh = min(self.car_battery_capacity_uWh, CAR_BATTERY_CAPACITY_uWh)
      if now - self.last_save_time >= 10:
        self.params.put("CarBatteryCapacity", int(self.car_battery_capacity_uWh))
        self.last_save_time = now

      # First measurement, set integration time
      with self.integration_lock:
        if self.last_measurement_time is None:
          self.last_measurement_time = now
          return

      if ignition:
        # If there is ignition, we integrate the charging rate of the car
        self.external_charging_since = None  # alternator voltage, then surface charge: not an external charger
        with self.integration_lock:
          self.power_used_uWh = 0
          integration_time_h = (now - self.last_measurement_time) / 3600
          if integration_time_h < 0:
            raise ValueError(f"Negative integration time: {integration_time_h}h")
          self.car_battery_capacity_uWh += (CAR_CHARGING_RATE_W * 1e6 * integration_time_h)
          self.last_measurement_time = now
      else:
        # Get current power draw somehow
        current_power = HARDWARE.get_current_power_draw()

        # Do the integration
        self._perform_integration(now, current_power, self._external_charging(now))
    except Exception:
      cloudlog.exception("Power monitoring calculation failed")

  def _external_charging(self, now: float) -> bool:
    """Ignition off and a charger holding the car battery voltage up for EXTERNAL_CHARGING_MIN_TIME_S"""
    if self.car_voltage_mV < VBATT_EXTERNAL_CHARGING * 1e3:
      self.external_charging_since = None
      return False
    if self.external_charging_since is None:
      self.external_charging_since = now
    return now - self.external_charging_since >= EXTERNAL_CHARGING_MIN_TIME_S

  def _perform_integration(self, t: float, current_power: float, external_charging: bool = False) -> None:
    with self.integration_lock:
      try:
        if self.last_measurement_time:
          integration_time_h = (t - self.last_measurement_time) / 3600
          power_used = (current_power * 1000000) * integration_time_h
          if power_used < 0:
            raise ValueError(f"Negative power used! Integration time: {integration_time_h} h Current Power: {power_used} uWh")
          self.power_used_uWh += power_used
          if external_charging:
            self.car_battery_capacity_uWh += EXTERNAL_CHARGING_RATE_W * 1e6 * integration_time_h
          else:
            self.car_battery_capacity_uWh -= power_used
          self.last_measurement_time = t
      except Exception:
        cloudlog.exception("Integration failed")

  # Get the power usage
  def get_power_used(self) -> int:
    return int(self.power_used_uWh)

  def get_car_battery_capacity(self) -> int:
    return int(self.car_battery_capacity_uWh)

  # Max Time Offroad
  def max_time_offroad_exceeded(self, offroad_time):
    param = self.params.get("MaxTimeOffroad")  # minutes, 0 = no limit
    if param is not None and param >= 0:
      return 0 < param * 60 <= offroad_time
    return offroad_time > MAX_TIME_OFFROAD_S

  # See if we need to shutdown
  def should_shutdown(self, ignition: bool, in_car: bool, offroad_timestamp: float | None, started_seen: bool):
    if offroad_timestamp is None:
      return False

    now = time.monotonic()
    should_shutdown = False
    offroad_time = (now - offroad_timestamp)
    low_voltage_shutdown = (self.car_voltage_mV < (VBATT_PAUSE_CHARGING * 1e3) and
                            offroad_time > VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S)
    should_shutdown |= self.max_time_offroad_exceeded(offroad_time)
    should_shutdown |= low_voltage_shutdown
    should_shutdown |= (self.car_battery_capacity_uWh <= 0)
    should_shutdown &= not ignition
    should_shutdown &= (not self.params.get_bool("DisablePowerDown"))
    should_shutdown &= in_car
    should_shutdown &= offroad_time > DELAY_SHUTDOWN_TIME_S
    should_shutdown |= self.params.get_bool("ForcePowerDown")
    should_shutdown &= started_seen or (now > MIN_ON_TIME_S)
    return should_shutdown
