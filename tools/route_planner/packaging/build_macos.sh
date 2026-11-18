#!/bin/bash
# One command: builds "dist/Route Planner.app" next to this script. Run from anywhere.
#   tools/route_planner/packaging/build_macos.sh
# Needs the uavloc conda env with PyInstaller (`$PYTHON -m pip install pyinstaller`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/uavloc/bin/python}"
APP="$HERE/dist/Route Planner.app"
EXECUTABLE="$APP/Contents/MacOS/route-planner"

rm -rf "$HERE/build" "$HERE/dist"
"$PYTHON" -m PyInstaller --noconfirm --clean --distpath "$HERE/dist" --workpath "$HERE/build" "$HERE/route_planner.spec"
rm -rf "$HERE/build"                                   # intermediate files are several GB

# PyInstaller already ad-hoc signs each Mach-O on arm64; sign the bundle as a whole so it launches as one unit.
codesign --force --deep -s - "$APP"
codesign --verify --deep "$APP" && echo "ad-hoc signature ok"

du -sh "$APP"
"$EXECUTABLE" --self-test
