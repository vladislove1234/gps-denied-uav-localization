"""Google Map Tiles API: satellite XYZ tiles under a session token.

Native resolution over rural Ukraine is z19; higher zooms are the z19 picture
upsampled (pocs/google-satellite). Only tile requests are billed; sessions are
free, and one lasts about two weeks.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from uavloc.track import REPO_ROOT

TILES_API = "https://tile.googleapis.com"
API_KEY_VAR = "GOOGLE_MAPS_API_KEY"
ENV_FILE = REPO_ROOT / ".env"  # gitignored
SESSION_TIMEOUT_S = 30


def load_api_key(env_file: Path = ENV_FILE) -> str:
    """The key from the environment, else from the gitignored `.env`."""
    if os.environ.get(API_KEY_VAR):
        return os.environ[API_KEY_VAR]
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == API_KEY_VAR:
                return value.strip().strip('"')
    raise KeyError(f"{API_KEY_VAR} is not set and not in {env_file}")


def create_session(api_key: str, region: str = "UA") -> str:
    """A satellite session token for `/v1/2dtiles` requests."""
    body = json.dumps({"mapType": "satellite", "language": "en-US", "region": region}).encode()
    request = urllib.request.Request(f"{TILES_API}/v1/createSession?key={api_key}", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=SESSION_TIMEOUT_S) as response:
        return json.load(response)["session"]


def satellite_tile_template(api_key: str, session: str) -> str:
    """Tile URL with `{z}/{x}/{y}` placeholders, the form XYZ tile fetchers take."""
    return f"{TILES_API}/v1/2dtiles/{{z}}/{{x}}/{{y}}?session={session}&key={api_key}"
