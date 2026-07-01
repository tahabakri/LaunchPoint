"""LaunchPoint — Controller-Origin Finder.

Estimate where a drone's operator is located from one or more ground-position
sightings of the drone, by exploiting the reciprocity of line-of-sight:
every place the drone could see is a place the controller could be.

Public API (Phases 0-5):

    from launchpoint import Sighting, find_origin
    from launchpoint.synthetic import SyntheticScenario

The heatmap is probabilistic; ~100% accuracy is neither expected nor claimed.
"""

from launchpoint.core.sighting import Sighting
from launchpoint.core.grid import RasterGrid
from launchpoint.config import Config
from launchpoint.coverage import CoverageEstimate, FlightArea

__all__ = [
    "Sighting",
    "RasterGrid",
    "Config",
    "CoverageEstimate",
    "FlightArea",
    "find_origin",
    "plan_launch_area",
    "__version__",
]

__version__ = "0.5.0"


def find_origin(*args, **kwargs):
    """Lazy import wrapper for the high-level pipeline (see launchpoint.pipeline)."""
    from launchpoint.pipeline import find_origin as _find_origin

    return _find_origin(*args, **kwargs)


def plan_launch_area(*args, **kwargs):
    """Lazy import wrapper for the reverse coverage planner."""
    from launchpoint.coverage import plan_launch_area as _plan_launch_area

    return _plan_launch_area(*args, **kwargs)
