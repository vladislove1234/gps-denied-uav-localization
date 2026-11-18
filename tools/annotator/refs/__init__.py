"""Reference sources. Importing this package registers every adapter."""

from .base import REGISTRY, Adapter, Location, RefResult, register  # noqa: F401
from . import osm, ms_footprints, worldcover, landcoverai, flair  # noqa: F401,E402
