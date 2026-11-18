# PyInstaller spec for the route planner. Build with build_macos.sh (it sets the output folders).
from pathlib import Path

import certifi
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SRC_ROOT = Path(SPECPATH).resolve().parents[2]           # .../src : holds uavloc/ and tools/
APP_NAME = "Route Planner"
EXECUTABLE_NAME = "route-planner"
BUNDLE_ID = "ua.uavloc.routeplanner"
APP_VERSION = "0.1.0"

# (source, destination folder inside the bundle). Destinations mirror the repo layout, so the app's
# `Path(__file__).parent / "web"` and `Path(__file__).parent / "lua"` keep working, and so do the model defaults
# in tools/route_planner/config.py (RESOURCE_ROOT = sys._MEIPASS).
DATAS = [
    (SRC_ROOT / "tools/route_planner/web", "tools/route_planner/web"),
    (SRC_ROOT / "uavloc/ardupilot/lua", "uavloc/ardupilot/lua"),
    (SRC_ROOT / "pocs/lraspp-pi5/models/lraspp-satellite.onnx", "pocs/lraspp-pi5/models"),
    (SRC_ROOT / "pocs/lraspp-pi5/models/lraspp-drone.onnx", "pocs/lraspp-pi5/models"),
    # conda-forge's certifi ships no cacert.pem (it points at $CONDA_PREFIX/ssl), so the hook alone bundles none
    (Path(certifi.where()), "certifi"),
]

# Not needed at run time. torch, torchvision, segmentation_models_pytorch, timm, pandas and matplotlib must stay:
# uavloc.planning imports them transitively.
EXCLUDES = [
    "ultralytics", "ultralytics_thop", "tensorboard", "tensorboard_data_server", "tensorflow",
    "IPython", "ipykernel", "ipywidgets", "jupyter", "jupyter_client", "jupyter_core", "notebook", "nbformat", "nbconvert",
    "papermill", "pytest", "_pytest", "tkinter", "PyQt5", "PyQt6", "PySide2",
    "torch.utils.tensorboard", "caffe2",
    "sphinx", "docutils", "black", "pylint", "mypy", "setuptools", "pip",
]

analysis = Analysis(
    [str(Path(SPECPATH) / "launcher.py")],
    pathex=[str(SRC_ROOT)],
    datas=[(str(source), destination) for source, destination in DATAS] + collect_data_files("rasterio"),   # gdal_data, proj_data
    hiddenimports=[
        # The self-test and the app import these by name, or lazily, where static analysis cannot see them.
        *collect_submodules("uavloc.planning"), *collect_submodules("uavloc.ardupilot"),
        *collect_submodules("uavloc.onboard"), *collect_submodules("rasterio"), "certifi",
    ],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz, analysis.scripts, [],
    exclude_binaries=True,
    name=EXECUTABLE_NAME,
    console=False,
    argv_emulation=False,
)
collected = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name=APP_NAME)
app = BUNDLE(
    collected,
    name=f"{APP_NAME}.app",
    bundle_identifier=BUNDLE_ID,
    version=APP_VERSION,
    info_plist={
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleShortVersionString": APP_VERSION,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "NSLocalNetworkUsageDescription": "Connects to an ArduPilot vehicle or SITL over the network.",
    },
)
