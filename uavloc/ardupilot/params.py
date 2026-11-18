"""The one ArduCopter parameter set: our position as GPS 2, a source set that flies on it, failsafes.

Every name and enum value below was checked against ArduPilot master (4.8-dev, October 2026):
`libraries/AP_GPS/AP_GPS.cpp`, `AP_GPS_Params.cpp`, `libraries/AP_NavEKF/AP_NavEKF_Source.cpp`,
`libraries/AP_BattMonitor/AP_BattMonitor_Params.cpp`, `libraries/RC_Channel/RC_Channel.h`,
`libraries/AP_SerialManager/AP_SerialManager.cpp`, `ArduCopter/Parameters.cpp`.
"""

from __future__ import annotations

from pathlib import Path

GPS_TYPE_MAV = 14                  # AP_GPS.h GPS_TYPE_MAV
GPS_AUTO_SWITCH_USE_PRIMARY = 0    # AP_GPS.cpp GPS_AUTO_SWITCH: 0 = always use GPS_PRIMARY
GPS_PRIMARY_REAL = 0               # GPS_PRIMARY: 0 = first GPS (the real one), 1 = second (ours)
GPS2_DELAY_MS = 50.0               # link latency only, after forward propagation (docs/ardupilot-integration.md 3)

EKF_SOURCE_GPS_XY = 3              # AP_NavEKF_Source.cpp POSXY/VELXY: 3 = GPS
EKF_SOURCE_NONE = 0
EKF_SOURCE_BARO = 1                # POSZ: 1 = baro
EKF_SOURCE_COMPASS = 1             # YAW: 1 = compass
EKF_SOURCE_GPS_Z = 3               # VELZ: 3 = GPS

SERIAL_PROTOCOL_MAVLINK2 = 2
SERIAL_BAUD_921600 = 921           # AP_SerialManager: baud in thousands
RC_OPTION_SCRIPTING_1 = 300        # RC_Channel.h AUX_FUNC::SCRIPTING_1
SWITCH_RC_CHANNEL = 9

FS_ACTION_SMARTRTL_OR_LAND_BATTERY = 4   # BATT_FS_LOW_ACT / BATT_FS_CRT_ACT: 4 = SmartRTL or Land
FS_THR_SMARTRTL_OR_LAND = 5              # FS_THR_ENABLE: 5 = always SmartRTL or Land
FS_EKF_ACTION_LAND = 1                   # FS_EKF_ACTION: 1 = Land if the mode needs position

PARAM_VALUE_FORMAT = "g"


def vehicle_params(gps2_delay_ms: float = GPS2_DELAY_MS) -> dict[str, float]:
    return {
        "GPS2_TYPE": GPS_TYPE_MAV,
        "GPS2_DELAY_MS": gps2_delay_ms,
        "GPS_AUTO_SWITCH": GPS_AUTO_SWITCH_USE_PRIMARY,
        "GPS_PRIMARY": GPS_PRIMARY_REAL,
        # Set 1: normal GPS flight.
        "EK3_SRC1_POSXY": EKF_SOURCE_GPS_XY,
        "EK3_SRC1_VELXY": EKF_SOURCE_GPS_XY,
        "EK3_SRC1_POSZ": EKF_SOURCE_BARO,
        "EK3_SRC1_VELZ": EKF_SOURCE_GPS_Z,
        "EK3_SRC1_YAW": EKF_SOURCE_COMPASS,
        # Set 2: position from the primary GPS (ours after the switch) but no GPS velocity, because GPS_INPUT
        # carries none and the driver would otherwise report a velocity of zero.
        "EK3_SRC2_POSXY": EKF_SOURCE_GPS_XY,
        "EK3_SRC2_VELXY": EKF_SOURCE_NONE,
        "EK3_SRC2_POSZ": EKF_SOURCE_BARO,
        "EK3_SRC2_VELZ": EKF_SOURCE_NONE,
        "EK3_SRC2_YAW": EKF_SOURCE_COMPASS,
        "EK3_SRC_OPTIONS": 0,
        "SERIAL2_PROTOCOL": SERIAL_PROTOCOL_MAVLINK2,
        "SERIAL2_BAUD": SERIAL_BAUD_921600,
        f"RC{SWITCH_RC_CHANNEL}_OPTION": RC_OPTION_SCRIPTING_1,
        "SCR_ENABLE": 1,
        "BATT_FS_LOW_ACT": FS_ACTION_SMARTRTL_OR_LAND_BATTERY,
        "BATT_FS_CRT_ACT": FS_ACTION_SMARTRTL_OR_LAND_BATTERY,
        "FS_THR_ENABLE": FS_THR_SMARTRTL_OR_LAND,
        "FS_EKF_ACTION": FS_EKF_ACTION_LAND,
    }


def write_param_file(path: Path, params: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name},{value:{PARAM_VALUE_FORMAT}}\n" for name, value in params.items()))


def read_param_file(path: Path) -> dict[str, float]:
    params = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.replace("\t", ",").replace(" ", ",").partition(",")
        if not separator:
            raise ValueError(f"{path}: expected NAME,VALUE, got {line!r}")
        params[name] = float(value)
    return params
