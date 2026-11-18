"""PyInstaller entry point: the route planner app, plus headless self-checks of the frozen bundle.

    launcher.py                       open the app
    launcher.py --settings PATH       open the app with a settings file
    launcher.py --self-test           import, load models, find resources; print a JSON report; exit 0/1
    launcher.py --self-test-plan      also plan the cached Birky case (needs caches in the settings)
"""

from __future__ import annotations

import importlib
import json
import os
import ssl
import sys
import time
import traceback
from pathlib import Path

SELF_TEST_FLAG = "--self-test"
SELF_TEST_PLAN_FLAG = "--self-test-plan"
SETTINGS_FLAG = "--settings"
IMPORTED_MODULES = (
    "uavloc.planning.planner", "uavloc.planning.routes", "uavloc.planning.elevation",
    "uavloc.ardupilot.mission", "uavloc.ardupilot.params", "uavloc.ardupilot.link", "uavloc.ardupilot.scripts",
    "uavloc.onboard.package", "uavloc.onboard.package_writer", "uavloc.onboard.stub", "uavloc.onboard.runner",
    "uavloc.features.onnx_segmenter",
    "tools.route_planner.main_window", "tools.route_planner.export", "tools.route_planner.planning_job",
    "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineCore", "PySide6.QtWebChannel",
    "rasterio", "scipy.ndimage", "cv2", "pymavlink.mavutil", "torch", "segmentation_models_pytorch",
)
BUNDLED_CA_FILE = "certifi/cacert.pem"   # where route_planner.spec puts the CA bundle
BIRKY_START = (49.905, 23.885)      # (lat, lon)
BIRKY_END = (49.915, 23.925)
BIRKY_ALTITUDE_M = 100.0


def use_bundled_ca_certificates() -> None:
    """The conda Python's CA file is not in the bundle, so urllib would fail every HTTPS request (tiles, elevation).

    Not `certifi.where()`: conda-forge's certifi points at `sys.prefix/ssl`, which the bundle does not have.
    """
    if getattr(sys, "frozen", False):
        os.environ.setdefault("SSL_CERT_FILE", str(Path(sys._MEIPASS) / BUNDLED_CA_FILE))


def check_tls(_config) -> dict[str, int]:
    return {"ca_certificates_loaded": ssl.create_default_context().cert_store_stats()["x509_ca"]}


def check_imports() -> dict[str, str]:
    """Module name -> "ok" or the error; one broken import does not hide the others."""
    results = {}
    for name in IMPORTED_MODULES:
        try:
            importlib.import_module(name)
            results[name] = "ok"
        except Exception as error:      # noqa: BLE001 - the report must list every failure
            results[name] = f"{type(error).__name__}: {error}"
    return results


def load_config(argv: list[str]):
    from tools.route_planner.config import AppConfig
    if SETTINGS_FLAG in argv:
        return AppConfig.load(Path(argv[argv.index(SETTINGS_FLAG) + 1]))
    return AppConfig.defaults()


def check_resources(config) -> dict[str, bool]:
    from tools.route_planner.main_window import WEB_DIR
    from uavloc.ardupilot.scripts import lua_script_path
    required = {"web/index.html": WEB_DIR / "index.html", "web/app.js": WEB_DIR / "app.js",
                "web/style.css": WEB_DIR / "style.css", "web/vendor/leaflet.js": WEB_DIR / "vendor/leaflet.js",
                "web/vendor/leaflet.css": WEB_DIR / "vendor/leaflet.css",
                "vision-gps-switch.lua": lua_script_path(),
                "satellite_onnx": config.satellite_onnx, "drone_onnx": config.drone_onnx}
    return {name: path.is_file() for name, path in required.items()}


def check_models(config) -> dict[str, dict]:
    import onnxruntime
    from uavloc.features.onnx_segmenter import OnnxSegmenter
    satellite = OnnxSegmenter(config.satellite_onnx)
    drone = onnxruntime.InferenceSession(str(config.drone_onnx), providers=["CPUExecutionProvider"])   # fixed non-square input
    return {"satellite": {"model_id": satellite.model_id, "n_classes": satellite.n_classes, "tile_px": satellite.tile_px},
            "drone": {"input_shape": drone.get_inputs()[0].shape, "output_shape": drone.get_outputs()[0].shape}}


def check_plan(config) -> dict:
    from tools.route_planner.planning_job import PlannerEngine
    from uavloc.geo import LatLon
    from uavloc.planning.config import MissionInputs
    started = time.monotonic()
    outcome = PlannerEngine(config).plan(MissionInputs(LatLon(*BIRKY_START), LatLon(*BIRKY_END), BIRKY_ALTITUDE_M))
    return {"routes": len(outcome.result.routes), "seconds": round(time.monotonic() - started, 1),
            "elevation_error": outcome.elevation_error,
            "class_overlay_bytes": len(outcome.class_overlay.data_url)}


def self_test(argv: list[str], plan: bool) -> int:
    report: dict = {"frozen": bool(getattr(sys, "frozen", False)), "python": sys.version.split()[0]}
    failures: list[str] = []
    steps = [("imports", lambda _: check_imports()), ("resources", check_resources), ("models", check_models), ("tls", check_tls)]
    config = None
    try:
        config = load_config(argv)
        report["config"] = {key: str(value) for key, value in vars(config).items() if key.endswith(("cache", "onnx", "dir"))}
    except Exception:       # noqa: BLE001
        failures.append("config")
        report["config_error"] = traceback.format_exc()
    if plan:
        steps.append(("plan", check_plan))
    for name, step in steps:
        if config is None:
            break
        try:
            report[name] = step(config)
        except Exception:   # noqa: BLE001
            failures.append(name)
            report[f"{name}_error"] = traceback.format_exc()
    if report.get("tls", {}).get("ca_certificates_loaded") == 0:
        failures.append("no CA certificates")
    failures += [f"import {m}" for m, status in report.get("imports", {}).items() if status != "ok"]
    failures += [f"missing {n}" for n, found in report.get("resources", {}).items() if not found]
    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main(argv: list[str]) -> int:
    use_bundled_ca_certificates()
    if SELF_TEST_PLAN_FLAG in argv:
        return self_test(argv, plan=True)
    if SELF_TEST_FLAG in argv:
        return self_test(argv, plan=False)
    from tools.route_planner.__main__ import main as app_main
    return app_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
