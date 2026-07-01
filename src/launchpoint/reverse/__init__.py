"""Launch-coverage search: the reverse problem to ``fusion``.

Given a target flight zone (a drawn circle + flight altitude), find candidate
launch/operator locations scored by how well each covers the *entire* zone.
"""

from launchpoint.reverse.pipeline import find_launch_area
from launchpoint.reverse.search import LaunchCoverageResult, LaunchSearchConfig

__all__ = ["find_launch_area", "LaunchCoverageResult", "LaunchSearchConfig"]
