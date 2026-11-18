"""Route planner entry point. Run from `src/` in the `uavloc` env.

    python -m tools.route_planner [--settings PATH]

Settings are read from `settings.json` in the user's app-data folder unless `--settings` says otherwise.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QStandardPaths, Qt
from PySide6.QtWidgets import QApplication

from tools.route_planner.config import SETTINGS_FILE_NAME, AppConfig
from tools.route_planner.main_window import MainWindow

ORGANIZATION_NAME = "uav-localization"
APPLICATION_NAME = "route-planner"
LOG_FILE_NAME = "route-planner.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(log_path: Path) -> None:
    """A packaged app has no visible stderr, so the log also goes to a file in the app-data folder."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")])


def default_settings_path() -> Path:
    return Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / SETTINGS_FILE_NAME


def create_application(argv: list[str]) -> QApplication:
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)   # QtWebEngine needs it before the app exists
    app = QApplication(argv)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationName(APPLICATION_NAME)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.route_planner")
    parser.add_argument("--settings", type=Path, default=None, help="JSON settings file (default: app-data folder)")
    args, qt_args = parser.parse_known_args(argv)
    app = create_application([sys.argv[0], *qt_args])
    configure_logging(Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)) / LOG_FILE_NAME)
    window = MainWindow(AppConfig.load(args.settings or default_settings_path()))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
