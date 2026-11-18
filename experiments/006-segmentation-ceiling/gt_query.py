"""Ground-truth class maps for 006. The code moved to `uavloc.matching.oracle` on its second use.

Kept so 006's notebooks import unchanged and its logged runs stay reproducible.
"""

from __future__ import annotations

from uavloc.matching.oracle import (  # noqa: F401
    ANNOTATED_DATE,
    DAMAGES,
    LABEL_SOURCES,
    MIN_OFFSET_PEAK_Z,
    OSM_TO_MODEL,
    QueryCorrelator,
    QueryFn,
    boundary_damage,
    damaged,
    gt_labels,
    gt_query,
    one_hot,
    osm_labels,
    stride_damage,
    substituted,
)
