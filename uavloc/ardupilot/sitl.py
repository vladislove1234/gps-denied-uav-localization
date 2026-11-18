"""Start and stop a native ArduPilot SITL copter. The ArduPilot checkout is passed in, never assumed."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from uavloc.ardupilot.params import write_param_file
from uavloc.ardupilot.scripts import lua_script_path
from uavloc.geo import LatLon

ARDUPILOT_DIR_ENV = "ARDUPILOT_DIR"
DEFAULT_ARDUPILOT_DIR = Path("~/Documents/Projects/ardupilot")
SITL_BINARY = Path("build/sitl/bin/arducopter")
COPTER_DEFAULTS = Path("Tools/autotest/default_params/copter.parm")
SITL_HOST = "127.0.0.1"
SITL_PORT = 5760                     # serial 0 of instance 0: the test harness
SITL_TELEM2_PORT = 5763              # serial 2, the port the Pi's UART (TELEM2) stands for: the onboard runner
STARTUP_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.2
EXTRA_PARAMS_NAME = "extra.parm"
SCRIPTS_DIR_NAME = "scripts"
LOG_NAME = "sitl.log"


@dataclass(frozen=True)
class SitlConfig:
    ardupilot_dir: Path
    home: LatLon
    home_alt_m: float
    heading_deg: float
    speedup: float
    extra_params: dict[str, float] = field(default_factory=dict)
    install_lua: bool = True
    work_dir: Path | None = None


def default_ardupilot_dir() -> Path:
    return Path(os.environ.get(ARDUPILOT_DIR_ENV, DEFAULT_ARDUPILOT_DIR)).expanduser()


def sitl_binary(ardupilot_dir: Path) -> Path:
    binary = ardupilot_dir / SITL_BINARY
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found: build it with `./waf configure --board sitl && ./waf copter`")
    return binary


def sitl_command(config: SitlConfig, extra_params_path: Path) -> list[str]:
    defaults = f"{config.ardupilot_dir / COPTER_DEFAULTS},{extra_params_path}"
    home = f"{config.home.lat},{config.home.lon},{config.home_alt_m},{config.heading_deg}"
    return [str(sitl_binary(config.ardupilot_dir)), "--model", "+", "--speedup", f"{config.speedup:g}",
            "--defaults", defaults, "-I0", "--home", home, "--wipe"]


def telem2_url() -> str:
    """Second MAVLink client of the same SITL: SITL serves one TCP client per serial port."""
    return f"tcp:{SITL_HOST}:{SITL_TELEM2_PORT}"


def port_is_open() -> bool:
    with socket.socket() as probe:
        probe.settimeout(POLL_INTERVAL_S)
        return probe.connect_ex((SITL_HOST, SITL_PORT)) == 0


def wait_for_port(process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SITL exited with code {process.returncode} during start-up")
        if port_is_open():
            return
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"SITL did not open port {SITL_PORT} within {timeout_s:g} s")


def stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@contextmanager
def start_sitl(config: SitlConfig) -> Iterator[str]:
    """Yield the MAVLink URL of a fresh SITL copter; the process is killed on exit."""
    if port_is_open():
        raise RuntimeError(f"port {SITL_PORT} is already in use: a SITL from an earlier run is still alive")
    work_dir = config.work_dir or Path(tempfile.mkdtemp(prefix="sitl-"))
    work_dir.mkdir(parents=True, exist_ok=True)
    extra_params_path = work_dir / EXTRA_PARAMS_NAME
    write_param_file(extra_params_path, config.extra_params)
    if config.install_lua:
        scripts_dir = work_dir / SCRIPTS_DIR_NAME
        scripts_dir.mkdir(exist_ok=True)
        shutil.copy(lua_script_path(), scripts_dir)
    with open(work_dir / LOG_NAME, "wb") as log:
        process = subprocess.Popen(sitl_command(config, extra_params_path), cwd=work_dir,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_for_port(process, STARTUP_TIMEOUT_S)
            yield f"tcp:{SITL_HOST}:{SITL_PORT}"
        finally:
            stop(process)
