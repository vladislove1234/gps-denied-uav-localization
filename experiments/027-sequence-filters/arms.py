"""027 imports its arms from `uavloc.simulation.arms` (promoted when 028 reused them), restricted to 027's five."""

from uavloc.simulation.arms import FilterParams, HeadSpec, build_correlator, estimator_factory, gated  # noqa: F401

ARMS = ("ins", "replace", "gated", "particles", "hypotheses")
FILTERS = ("particles", "hypotheses")
