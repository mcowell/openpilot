from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params
from openpilot.system.hardware.power_monitoring import PowerMonitoring, CAR_BATTERY_CAPACITY_uWh, \
  CAR_CHARGING_RATE_W, VBATT_PAUSE_CHARGING, DELAY_SHUTDOWN_TIME_S, MAX_TIME_OFFROAD_S, VBATT_EXTERNAL_CHARGING, \
  EXTERNAL_CHARGING_MIN_TIME_S, EXTERNAL_CHARGING_RATE_W

# Create fake time
ssb = 0.
def mock_time_monotonic():
  global ssb
  ssb += 1.
  return ssb

def set_mock_time(value):
  global ssb
  ssb = value

TEST_DURATION_S = 50
GOOD_VOLTAGE = 12 * 1e3
VOLTAGE_BELOW_PAUSE_CHARGING = (VBATT_PAUSE_CHARGING - 1) * 1e3

def pm_patch(mocker, name, value, constant=False):
  if constant:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", value)
  else:
    mocker.patch(f"openpilot.system.hardware.power_monitoring.{name}", return_value=value)


class TestPowerMonitoring(OpenpilotTestCase):
  def setup_method(self):
    self._fixture("mocker").patch("time.monotonic", mock_time_monotonic)
    self.params = Params()

  # Test to see that it doesn't do anything when pandaState is None
  def test_panda_state_present(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(None, False)
    assert pm.get_power_used() == 0
    assert pm.get_car_battery_capacity() == (CAR_BATTERY_CAPACITY_uWh / 10)

  # Test to see that it doesn't integrate offroad when ignition is True
  def test_offroad_ignition(self):
    pm = PowerMonitoring()
    for _ in range(10):
      pm.calculate(GOOD_VOLTAGE, True)
    assert pm.get_power_used() == 0

  # Test to see that it integrates with discharging battery
  def test_offroad_integration_discharging(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_power_usage = ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_power_used() - expected_power_usage) < 10

  # Test to check positive integration of car_battery_capacity
  def test_car_battery_integration_onroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    expected_capacity = ((TEST_DURATION_S/3600) * CAR_CHARGING_RATE_W * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check positive integration upper limit
  def test_car_battery_integration_upper_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh - 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, True)
    estimated_capacity = CAR_BATTERY_CAPACITY_uWh + (CAR_CHARGING_RATE_W / 3600 * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Test to check negative integration of car_battery_capacity
  def test_car_battery_integration_offroad(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    expected_capacity = CAR_BATTERY_CAPACITY_uWh - ((TEST_DURATION_S/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - expected_capacity) < 10

  # Test to check negative integration lower limit
  def test_car_battery_integration_lower_limit(self, mocker):
    POWER_DRAW = 4
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 1000
    for _ in range(TEST_DURATION_S + 1):
      pm.calculate(GOOD_VOLTAGE, False)
    estimated_capacity = 0 - ((1/3600) * POWER_DRAW * 1e6)
    assert abs(pm.get_car_battery_capacity() - estimated_capacity) < 10

  # Battery charger / maintainer on the car battery with the ignition off: the capacity budget refills
  def test_external_charger_refills_capacity(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 4)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh / 10
    charger_v = 14.3e3
    for _ in range(EXTERNAL_CHARGING_MIN_TIME_S + 3600):
      pm.calculate(charger_v, False)
    assert pm.external_charging_since is not None
    # less than an hour of refill: the LPF voltage takes a while to rise, then it must stay up EXTERNAL_CHARGING_MIN_TIME_S
    assert CAR_BATTERY_CAPACITY_uWh / 10 + 0.5 * EXTERNAL_CHARGING_RATE_W * 1e6 < pm.get_car_battery_capacity() < \
           CAR_BATTERY_CAPACITY_uWh / 10 + EXTERNAL_CHARGING_RATE_W * 1e6
    assert pm.get_power_used() > 0  # the device's own draw is still reported
    assert not pm.should_shutdown(False, True, ssb - 2 * DELAY_SHUTDOWN_TIME_S, True)

  # A short time above the threshold (surface charge after a drive) doesn't count as charging
  def test_brief_high_voltage_drains(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 4)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh / 2
    pm.car_voltage_mV = VBATT_EXTERNAL_CHARGING * 1e3 + 300
    for _ in range(EXTERNAL_CHARGING_MIN_TIME_S - 10):
      pm.calculate(VBATT_EXTERNAL_CHARGING * 1e3 + 300, False)
    assert pm.get_car_battery_capacity() < CAR_BATTERY_CAPACITY_uWh / 2
    # and dropping below the threshold restarts the wait
    for _ in range(200):
      pm.calculate(12.5e3, False)
    assert pm.external_charging_since is None

  # High voltage while the ignition is on is the alternator: after ignition off, the wait starts over
  def test_ignition_resets_external_charging(self, mocker):
    pm_patch(mocker, "HARDWARE.get_current_power_draw", 4)
    pm = PowerMonitoring()
    pm.car_voltage_mV = 14.3e3
    for _ in range(2 * EXTERNAL_CHARGING_MIN_TIME_S):
      pm.calculate(14.3e3, True)
    before = pm.get_car_battery_capacity()
    for _ in range(EXTERNAL_CHARGING_MIN_TIME_S - 10):
      pm.calculate(14.3e3, False)
    assert pm.get_car_battery_capacity() < before

  # Test to check policy of stopping charging after MAX_TIME_OFFROAD_S
  def test_max_time_offroad(self, mocker):
    MOCKED_MAX_OFFROAD_TIME = 3600
    POWER_DRAW = 0 # To stop shutting down for other reasons
    pm_patch(mocker, "MAX_TIME_OFFROAD_S", MOCKED_MAX_OFFROAD_TIME, constant=True)
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    start_time = ssb
    ignition = False
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME - 1)
    assert not pm.should_shutdown(ignition, True, start_time, False)
    set_mock_time(start_time + MOCKED_MAX_OFFROAD_TIME)
    assert pm.should_shutdown(ignition, True, start_time, False)

  def test_car_voltage(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 350
    VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S = 50
    pm_patch(mocker, "VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S", VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S, constant=True)
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    ignition = False
    start_time = ssb
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert pm.should_shutdown(ignition, True, start_time, True) == \
                          (pm.car_voltage_mV < VBATT_PAUSE_CHARGING * 1e3 and \
                          (ssb - start_time) > VOLTAGE_SHUTDOWN_MIN_OFFROAD_TIME_S and \
                            (ssb - start_time) > DELAY_SHUTDOWN_TIME_S)
    assert pm.should_shutdown(ignition, True, start_time, True)

  # Test to check policy of not stopping charging when DisablePowerDown is set
  def test_disable_power_down(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    self.params.put_bool("DisablePowerDown", True, block=True)
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    ignition = False
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, True, ssb, False)
    assert not pm.should_shutdown(ignition, True, ssb, False)

  # Test to check policy of not stopping charging when ignition
  def test_ignition(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh
    ignition = True
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, True, ssb, False)
    assert not pm.should_shutdown(ignition, True, ssb, False)

  # Test to check policy of not stopping charging when harness is not connected
  def test_harness_connection(self, mocker):
    POWER_DRAW = 0 # To stop shutting down for other reasons
    TEST_TIME = 100
    pm_patch(mocker, "HARDWARE.get_current_power_draw", POWER_DRAW)
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = CAR_BATTERY_CAPACITY_uWh

    ignition = False
    for i in range(TEST_TIME):
      pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)
      if i % 10 == 0:
        assert not pm.should_shutdown(ignition, False, ssb, False)
    assert not pm.should_shutdown(ignition, False, ssb, False)

  def test_delay_shutdown_time(self):
    pm = PowerMonitoring()
    pm.car_battery_capacity_uWh = 0
    ignition = False
    in_car = True
    offroad_timestamp = ssb
    started_seen = True
    pm.calculate(VOLTAGE_BELOW_PAUSE_CHARGING, ignition)

    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S - 1)
    assert not pm.should_shutdown(ignition, in_car, offroad_timestamp, started_seen), \
                     f"Should not shutdown before {DELAY_SHUTDOWN_TIME_S} seconds offroad time"
    set_mock_time(offroad_timestamp + DELAY_SHUTDOWN_TIME_S)
    assert pm.should_shutdown(ignition, in_car,
                                       offroad_timestamp,
                                       started_seen), \
                    f"Should shutdown after {DELAY_SHUTDOWN_TIME_S} seconds offroad time"

  @parameterized.expand(
    [
      # No max time set – fallback to default (30 hours)
      (None, 0, False),
      (None, MAX_TIME_OFFROAD_S + 1, True),  # exceeds 30h (1800+ mins)

      # Valid max time values (in minutes)
      (60, 59, False),  # under limit
      (60, 120, True),  # over limit
      (10, 8, False),  # under limit
      (10, 11, True),  # over limit

      # Edge case: max time is zero → no limit enforced
      (0, 0, False),
      (0, 400, False),

      # Invalid max time formats or negative values → fallback to 30 hours
      (-100, 100, False),  # should fallback to 30h
      (-1, MAX_TIME_OFFROAD_S + 1, True),  # should fallback to 30h, and exceed it
    ]
  )
  def test_max_time_offroad_exceeded(self, max_time_offroad, offroad_time_min, expected_result):
    # Set the parameter if provided
    if max_time_offroad is not None:
      self.params.put("MaxTimeOffroad", max_time_offroad, block=True)

    # Convert offroad time from minutes to seconds
    offroad_time_s = offroad_time_min * 60

    pm = PowerMonitoring()
    result = pm.max_time_offroad_exceeded(offroad_time_s)

    assert result == expected_result
