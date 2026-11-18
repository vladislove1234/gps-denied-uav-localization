from __future__ import annotations

import time
from pathlib import Path

import pytest

from chain_support import build_package
from uavloc.ardupilot.sitl import port_is_open

SITL_PORT_WAIT_S = 120.0             # another agent may be using the one SITL port; wait, then skip
SITL_PORT_POLL_S = 2.0


@pytest.fixture
def free_sitl_port() -> None:
    deadline = time.monotonic() + SITL_PORT_WAIT_S
    while port_is_open():
        if time.monotonic() > deadline:
            pytest.skip("SITL port 5760 stayed busy")
        time.sleep(SITL_PORT_POLL_S)


@pytest.fixture
def package_zip(tmp_path: Path) -> Path:
    return build_package(tmp_path)
