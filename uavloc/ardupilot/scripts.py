"""Where the packaged Lua script lives."""

from __future__ import annotations

from pathlib import Path

LUA_DIR = Path(__file__).resolve().parent / "lua"
SWITCH_SCRIPT_NAME = "vision-gps-switch.lua"


def lua_script_path() -> Path:
    return LUA_DIR / SWITCH_SCRIPT_NAME
