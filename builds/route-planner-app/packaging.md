# route-planner-app: macOS `.app` packaging

**Status:** builds, frozen self-test passes, GUI window unverified (sandbox) · **Step 5 of** `docs/route-planner.md`

## Build

```
tools/route_planner/packaging/build_macos.sh
```

Needs the `uavloc` env with PyInstaller (`~/miniconda3/envs/uavloc/bin/python -m pip install pyinstaller`; 6.22.3 used).
The script runs PyInstaller with `route_planner.spec`, deletes its intermediate `build/` folder, ad-hoc signs the bundle
(`codesign --force --deep -s -`), verifies the signature and runs `--self-test`. Takes about 5 minutes on an idle M1.
Output: `tools/route_planner/packaging/dist/Route Planner.app` (arm64, bundle id `ua.uavloc.routeplanner`,
windowed, unsigned apart from the ad-hoc signature). Size: **1.14 GiB**. `build/` and `dist/` are gitignored.
Never run two builds at once: they share PyInstaller's cache and the second corrupts the first.

| File | Job |
|---|---|
| `route_planner.spec` | what to bundle and exclude |
| `launcher.py` | entry point: the app, `--self-test`, `--self-test-plan`; sets the bundled CA file |
| `build_macos.sh` | the one build command |

## What is inside

- `tools/route_planner/web/` (with vendored Leaflet), `uavloc/ardupilot/lua/vision-gps-switch.lua`,
  `pocs/lraspp-pi5/models/lraspp-satellite.onnx` and `lraspp-drone.onnx`, `rasterio` `gdal_data`/`proj_data`.
  Destinations mirror the repo layout, so `Path(__file__)` lookups in the app and `RESOURCE_ROOT` in `config.py`
  (`sys._MEIPASS` when frozen) find them. `tools/annotator/classes.json` is not bundled: the app never reads it.
- QtWebEngine helper and resources come from the PySide6 hooks (QtWebEngineCore framework, 292 MB of the total).
- Size split: PySide6/Qt 470 MB, torch 203 MB, cv2 87 MB, onnxruntime 70 MB, rasterio 48 MB, scipy 38 MB.

## Paths when frozen (`tools/route_planner/config.py`)

Read-only resources come from the bundle (`Contents/Frameworks/...`). Tile cache, planner cache and exports default to
`~/Library/Application Support/uav-localization/route-planner/{tile-cache,planner-cache,route-planner-exports}`.
`settings.json` in that folder (or `--settings PATH`) overrides any of them; relative paths in it resolve against the
bundle, so use absolute ones. Run from source nothing changes.

## Exclusions

`ultralytics`, `tensorboard`, `tensorflow`, IPython/Jupyter family, `papermill`, `pytest`, `tkinter`, PyQt5/6, PySide2,
`torch.utils.tensorboard`, sphinx/docutils, black/pylint/mypy, setuptools, pip. Each was checked by the frozen import
check. Two exclusions were tried and had to be removed: `torch.distributed` and `torch.testing._internal` (torch and
timm import both at load time). Kept because `uavloc.planning` pulls them in: torch, torchvision,
segmentation_models_pytorch, timm, pandas, matplotlib.

## Verified

Frozen, on this Mac (`Route Planner.app/Contents/MacOS/route-planner`):

- `--self-test`: exit 0, `"ok": true`. 24 modules import (planning, ardupilot, onboard, the app modules, QtWebEngine
  Widgets/Core/WebChannel, rasterio, scipy, cv2, pymavlink, torch, smp). `web/` files, the Lua script and both `.onnx`
  files are found. `OnnxSegmenter` loads the satellite model (id `f265dcd9e266`, 6 classes, 512 px tile); the drone
  model opens in onnxruntime (input `1x3x416x736`, a non-square fixed size `OnnxSegmenter` does not support, so the
  check opens it directly). 121 CA certificates loaded.
- `--self-test-plan --settings <settings pointing at data/derived/{tile,planner}-cache>`: Birky case
  (49.905, 23.885 -> 49.915, 23.925) gives 2 routes, same count as from source; 11 s with the cache warm, elevation
  check from Open-Meteo succeeded, class overlay rendered.
- `open -n "Route Planner.app"`: LaunchServices accepts the bundle id, the process stays up, no stdout/stderr output
  besides the matplotlib font-cache message.
- `tests/route_planner`: 24 passed with the `config.py` change.

## Not verified

- **The GUI window and QtWebEngine inside the bundle.** In the agent sandbox the app has no WindowServer connection
  (`lsappinfo` shows `!cgsConnection`) and no `QtWebEngineProcess` child ever starts, the same limit as in `notes.md`.
  So whether the map renders, `qwebchannel.js` loads from the `file://` page, and a renderer starts from the bundle
  is open. Do this once: double-click the app (first time: right-click, Open), click Start and End, Plan.
- Planner cache did not hit on the first source run (10 min recompute, then warm); the `--self-test-plan` timing above is
  warm.
- Ad-hoc signature only: no hardened runtime, no entitlements, no notarisation. Not tested on another Mac or Intel.
- A real flight controller over serial, the first launch downloading tiles, Windows.

## Known issues

- Unsigned: Gatekeeper blocks a copy that was downloaded or AirDropped; right-click, Open (or
  `xattr -dr com.apple.quarantine "Route Planner.app"`).
- Matplotlib rebuilds its font cache on every start (PyInstaller points `MPLCONFIGDIR` at a temp folder): a few
  seconds of start-up cost, no functional effect.
- 1.14 GiB, mostly torch + Qt WebEngine + OpenCV. Smaller needs lazy imports in `uavloc/` (torch only to read the
  matcher config) which this app does not own.
- The conda Python has no CA file in the bundle; `launcher.py` points `SSL_CERT_FILE` at certifi's. Without it every
  HTTPS call (tiles, elevation) fails with `CERTIFICATE_VERIFY_FAILED`. This only shows frozen, never from source.
- Windows: later. The spec is mostly portable; `build_macos.sh`, the codesign step and the app-data path would need a
  Windows twin.
