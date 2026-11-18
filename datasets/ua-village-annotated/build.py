"""Build `ua-village-annotated` -- the recipe is the annotator's export.

    python datasets/ua-village-annotated/build.py [--new | --stale | --all]

Everything that decides the bytes lives in ``tools/annotator/``: the class
registry (``classes.json``), the reference adapters (``refs/``) and the
composition (``server/export.py``). This file exists so the dataset folder
names its recipe like every other dataset here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.annotator.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main(["export", *(sys.argv[1:] or ["--new"])])
