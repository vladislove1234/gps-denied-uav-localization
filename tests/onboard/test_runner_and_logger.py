from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from onboard_fakes import (ORIGIN, FakeConnection, RecordingLogger, RecordingSink, ScriptedLocalizer, at, stream)
from uavloc.onboard import cli
from uavloc.onboard.cameras import NullCamera
from uavloc.onboard.contracts import Fix, FlightState, Frame
from uavloc.onboard.logger import JsonlFlightLogger
from uavloc.onboard.mavlink_sources import ArdupilotHeight
from uavloc.onboard.package import load_package
from uavloc.onboard.replay import replay_stub
from uavloc.onboard.runner import OnboardRunner, RunnerParts


class ScriptedStates:
    def __init__(self, states: list[FlightState | None]) -> None:
        self.states = list(states)
        self.asked_at: list[float] = []

    def latest(self) -> FlightState | None:
        return self.states.pop(0)

    def state_at(self, time_s: float) -> FlightState | None:
        self.asked_at.append(time_s)
        return self.states.pop(0)


class DoubleHeight:
    def height_m(self, state: FlightState) -> float:
        return state.altitude_m * 2


class OneFrameCamera:
    def __init__(self, frame: Frame) -> None:
        self.frames = [frame, None]

    def next_frame(self) -> Frame | None:
        return self.frames.pop(0)


def state(time_s: float) -> FlightState:
    return FlightState(time_s, ORIGIN, 40.0, 90.0, 0.0, 0.0, 0.0, 5.0)


def make_runner(package, camera, states, localizer, period_s: float = 0.5):
    sink, logger = RecordingSink(), RecordingLogger()
    parts = RunnerParts(camera, states, DoubleHeight(), localizer, sink, logger)
    ticks = {"t": 0.0, "slept": []}
    runner = OnboardRunner(package, parts, period_s, clock=lambda: ticks["t"],
                           sleep=lambda s: (ticks["slept"].append(s), ticks.__setitem__("t", ticks["t"] + s)))
    return runner, sink, logger, ticks


@pytest.fixture
def package(package_zip: Path):
    return load_package(package_zip)


def test_runner_runs_n_steps_sends_logs_and_closes(package) -> None:
    fixes = [Fix(float(k), at(300.0, 0.0), 6.0, 1.0, "stub") for k in range(4)]
    localizer = ScriptedLocalizer(fixes)
    runner, sink, logger, ticks = make_runner(package, NullCamera(), ScriptedStates([state(k) for k in range(4)]), localizer)
    assert runner.run(max_steps=4) == 4
    assert sink.fixes == logger.fixes == fixes
    assert len(logger.states) == 4 and logger.closed
    assert localizer.calls[0][1].altitude_m == 80.0       # the height source decides the altitude the localizer sees
    assert ticks["slept"] == [0.5] * 4                    # an instant step still takes one period


def test_runner_skips_a_step_without_state(package) -> None:
    runner, sink, logger, _ = make_runner(package, NullCamera(), ScriptedStates([None, state(1.0)]),
                                          ScriptedLocalizer([Fix(1.0, None, 6.0, 0.0, "blind")]))
    runner.run(max_steps=2)
    assert runner.steps_without_state == 1 and len(sink.fixes) == 1


def test_localizer_exception_becomes_a_no_fix(package) -> None:
    runner, sink, logger, _ = make_runner(package, NullCamera(), ScriptedStates([state(1.0)]),
                                          ScriptedLocalizer(error=RuntimeError("onnx died")))
    fix = runner.step()
    assert fix.position is None and fix.reason == "error: RuntimeError"
    assert sink.fixes == [fix] and logger.fixes == [fix]


def test_frame_selects_the_state_at_its_capture_time(package) -> None:
    frame = Frame(7.25, np.zeros((4, 4, 3), np.uint8))
    states = ScriptedStates([state(7.25)])
    runner, _, logger, _ = make_runner(package, OneFrameCamera(frame), states, ScriptedLocalizer([Fix(7.25, None, 6.0, 0.0, "x")]))
    runner.step()
    assert states.asked_at == [7.25] and logger.frames == [frame]


def test_runner_stops_on_request_and_still_closes(package) -> None:
    runner, _, logger, _ = make_runner(package, NullCamera(), ScriptedStates([state(0.0)]),
                                       ScriptedLocalizer([Fix(0.0, None, 6.0, 0.0, "x")]))
    runner.run(should_stop=lambda: runner.steps >= 1)
    assert runner.steps == 1 and logger.closed


def test_runner_rejects_a_non_positive_period(package) -> None:
    with pytest.raises(ValueError):
        make_runner(package, NullCamera(), ScriptedStates([]), ScriptedLocalizer(), period_s=0.0)


def test_logger_writes_lines_every_nth_frame_and_the_manifest(tmp_path: Path, package) -> None:
    logger = JsonlFlightLogger(tmp_path / "run", frame_every=5, manifest=package.manifest_path)
    logger.log_fix(Fix(1.0, at(1.0, 1.0), 6.0, 1.0, "stub"))
    logger.log_fix(Fix(2.0, None, 6.0, 0.0, "blind"))
    logger.log_state(state(1.0))
    for k in range(11):
        logger.log_frame(Frame(float(k), np.full((8, 8, 3), k * 20, np.uint8)))
    logger.close()
    fixes = [json.loads(line) for line in (tmp_path / "run" / "fixes.jsonl").read_text().splitlines()]
    assert fixes[0]["lat"] is not None and fixes[1] == {"time_s": 2.0, "lat": None, "lon": None, "uncertainty_m": 6.0,
                                                        "quality": 0.0, "reason": "blind"}
    saved = json.loads((tmp_path / "run" / "states.jsonl").read_text())
    assert saved["position"] == list(ORIGIN) and saved["velocity_east_mps"] == 5.0
    assert [p.name[:12] for p in sorted((tmp_path / "run" / "frames").iterdir())] == [
        "frame-000000", "frame-000005", "frame-000010"]
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["mission_id"] == "test"


def test_logger_rejects_a_zero_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JsonlFlightLogger(tmp_path / "run", frame_every=0)


def test_cli_requires_the_stub_and_a_package(tmp_path: Path) -> None:
    assert cli.main(["--packages", str(tmp_path), "--connection", "udp:x"]) == cli.EXIT_NO_LOCALIZER
    assert cli.main(["--packages", str(tmp_path), "--connection", "udp:x", "--stub"]) == cli.EXIT_NO_PACKAGE


@pytest.mark.parametrize(("east_m", "expected_sent", "reason"), [(300.0, 3, "stub"), (550.0, 0, "blind")])
def test_cli_end_to_end_against_a_fake_flight_controller(tmp_path: Path, package_zip: Path, monkeypatch, east_m: float,
                                                         expected_sent: int, reason: str) -> None:
    conn = FakeConnection()
    stream(conn, 0.0, 4.0, position=at(east_m, -300.0))
    monkeypatch.setattr(cli, "connect", lambda url, timeout_s: conn)
    code = cli.main(["--packages", str(package_zip.parent), "--connection", "udp:x", "--stub", "--log-dir",
                     str(tmp_path / "logs"), "--max-steps", "3", "--period-s", "0.01"])
    assert code == cli.EXIT_OK
    run = next((tmp_path / "logs").iterdir())
    fixes = [json.loads(line) for line in (run / "fixes.jsonl").read_text().splitlines()]
    assert [f["reason"] for f in fixes] == [reason] * 3
    assert len(conn.mav.gps_inputs) == expected_sent
    assert (run / "manifest.json").exists()
    assert {int(c[4]) for c in conn.mav.commands} >= {30, 32, 33, 164}


def test_ardupilot_height_is_the_state_altitude() -> None:
    assert ArdupilotHeight().height_m(state(0.0)) == 40.0


def test_replay_drops_fixes_exactly_over_the_blind_stretch(package) -> None:
    result = replay_stub(package, speed_mps=5.0, altitude_m=40.0, period_s=1.0, noise_m=2.0, seed=0)
    blind = result.dropped
    stretch_m = 200.0                                    # BLIND_FROM_EAST_M..BLIND_TO_EAST_M of the synthetic world
    assert result.length_m == pytest.approx(2000.0, rel=1e-3)
    assert blind.sum() * 5.0 == pytest.approx(2 * stretch_m, abs=30.0)       # out and back
    assert result.longest_dropped_run_m() == pytest.approx(stretch_m, abs=15.0)
    assert result.error_m().mean() == pytest.approx(2.0 * 1.25, abs=0.4)
