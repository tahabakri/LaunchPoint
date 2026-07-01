"""Synthetic scenarios for objective pass/fail validation.

Built *first* (Phase 0) so every later phase has ground truth: plant a fake
controller, synthesise a terrain surface, generate drone sightings that are
geometrically visible from that controller, then confirm the pipeline recovers
the controller's region.
"""

from launchpoint.synthetic.scenario import (
    LaunchScenario,
    SyntheticScenario,
    make_default_scenario,
    make_launch_scenario,
)

__all__ = [
    "SyntheticScenario",
    "make_default_scenario",
    "LaunchScenario",
    "make_launch_scenario",
]
