"""`python -m uavloc.onboard`: pick the newest mission package, connect to the flight controller, run the loop."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

from uavloc.onboard.cameras import NullCamera
from uavloc.onboard.logger import DEFAULT_FRAME_EVERY, JsonlFlightLogger
from uavloc.onboard.mavlink_sources import ArdupilotHeight, MavlinkFlightState, MavlinkRouter, SimTruth, connect
from uavloc.onboard.package import PackageError, load_package, newest_package
from uavloc.onboard.runner import DEFAULT_PERIOD_S, OnboardRunner, RunnerParts
from uavloc.onboard.sinks import GpsInputSink
from uavloc.onboard.stub import SITL_NOISE_M, StubLocalizer

DEFAULT_CONNECT_TIMEOUT_S = 30.0
DEFAULT_LOG_DIR = Path("flight-logs")
EXIT_OK = 0
EXIT_NO_PACKAGE = 2
EXIT_NO_LINK = 3
EXIT_NO_LOCALIZER = 4
EXIT_BAD_ARGUMENT = 5
RUN_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"

log = logging.getLogger("uavloc.onboard")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m uavloc.onboard", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--packages", type=Path, help="directory of mission packages; the newest valid one runs")
    source.add_argument("--package", type=Path, help="one mission package (zip or folder)")
    parser.add_argument("--connection", required=True,
                        help="pymavlink URL, e.g. /dev/ttyAMA0,921600 on the Pi or tcp:127.0.0.1:5760 in SITL")
    parser.add_argument("--stub", action="store_true", help="localize with the simulator truth plus noise (SITL only)")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--period-s", type=float, default=DEFAULT_PERIOD_S)
    parser.add_argument("--frame-every", type=int, default=DEFAULT_FRAME_EVERY)
    parser.add_argument("--noise-m", type=float, default=SITL_NOISE_M, help="stub: noise std per axis")
    parser.add_argument("--latency-s", type=float, default=0.0, help="stub: fix describes this long ago")
    parser.add_argument("--seed", type=int, default=0, help="stub noise seed")
    parser.add_argument("--no-blind", action="store_true", help="stub: also answer over blind stretches")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="SITL speedup: the period and the clocks then run in simulated seconds")
    parser.add_argument("--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps (tests)")
    return parser


def run_directory(log_dir: Path, mission_id: str) -> Path:
    return log_dir / f"run-{datetime.now(timezone.utc).strftime(RUN_STAMP_FORMAT)}-{mission_id}"


def scaled_clocks(speedup: float) -> tuple[Callable[[], float], Callable[[float], None]]:
    """`(clock, sleep)` in simulated seconds, for a simulator running `speedup` times faster than the wall clock."""
    return (lambda: time.monotonic() * speedup), (lambda duration_s: time.sleep(duration_s / speedup))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.stub:
        log.error("no real localizer yet: run with --stub (SITL)")
        return EXIT_NO_LOCALIZER
    try:
        path = args.package or newest_package(args.packages)
        package = load_package(path)
    except (FileNotFoundError, PackageError) as error:
        log.error("%s", error)
        return EXIT_NO_PACKAGE
    log.info("package %s (%s)", package.mission_id, path)
    try:
        conn = connect(args.connection, args.connect_timeout_s)
    except (TimeoutError, OSError) as error:
        log.error("cannot connect to %s: %s", args.connection, error)
        return EXIT_NO_LINK
    if args.speedup <= 0:
        log.error("--speedup must be positive, got %g", args.speedup)
        return EXIT_BAD_ARGUMENT
    clock, sleep = scaled_clocks(args.speedup)
    router = MavlinkRouter(conn, monotonic=clock)
    flight_state = MavlinkFlightState(router)
    parts = RunnerParts(
        camera=NullCamera(), state_source=flight_state, height=ArdupilotHeight(),
        localizer=StubLocalizer(SimTruth(router), args.noise_m, args.seed, args.latency_s, args.no_blind),
        sink=GpsInputSink(conn, motion=flight_state, altitude=flight_state),
        logger=JsonlFlightLogger(run_directory(args.log_dir, package.mission_id), args.frame_every,
                                 package.manifest_path))
    runner = OnboardRunner(package, parts, args.period_s, clock, sleep)
    stopping = {"requested": False}   # systemd stops the service with SIGTERM

    def request_stop(signum: int, frame: object) -> None:
        stopping["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    steps = runner.run(args.max_steps, lambda: stopping["requested"])
    sink = parts.sink
    log.info("stopped after %d steps (%d without state); fixes sent %d, no-fix %d, stale %d", steps,
             runner.steps_without_state, sink.sent, sink.skipped_no_fix, sink.skipped_stale)
    return EXIT_OK

