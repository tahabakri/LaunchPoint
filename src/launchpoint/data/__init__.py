"""Keyless data layer.

Every source here is free and requires no account or API key: data is pulled by
plain anonymous HTTPS (Cloud-Optimized GeoTIFF range requests) or the public
Overpass API. Phase 1 establishes the DSM pipeline; Phase 4 adds canopy and
buildings and derives the bare-earth surface.

The orchestrator ``build_surface_stack`` returns the three surfaces the analysis
needs: the occluding DSM, the bare-earth target surface, and a launch-feasibility
weight.
"""

from launchpoint.data.surface import SurfaceStack, build_surface_stack

__all__ = ["SurfaceStack", "build_surface_stack"]
