"""Earth-curvature and atmospheric-refraction correction.

Over long rays the Earth bulges up between observer and target, hiding distant
low ground; light refraction partially counteracts this. The standard surveying
approximation rolls both into an *effective* Earth radius k*R (k = 4/3), after
which the apparent drop of a point at horizontal distance d is::

    drop(d) = d^2 / (2 * k * R)

At 12 km this is ~8.5 m (vs ~11.3 m with no refraction) — enough that ignoring
it makes long rays "see" ground that is really below the horizon. We subtract drop(d) from the apparent height
of every intermediate terrain sample and of the target.
"""

from __future__ import annotations

import numpy as np

from launchpoint.config import EARTH_RADIUS_M, REFRACTION_K


def inv_two_effective_radius(refraction: bool = True, k: float = REFRACTION_K) -> float:
    """Return 1 / (2 * R_eff). Multiply by d^2 to get the curvature drop.

    If ``refraction`` is False, 0.0 is returned (curvature correction disabled).
    """
    if not refraction:
        return 0.0
    return 1.0 / (2.0 * k * EARTH_RADIUS_M)


def curvature_drop(distance_m, refraction: bool = True, k: float = REFRACTION_K):
    """Apparent vertical drop (metres) at horizontal distance ``distance_m``."""
    return np.asarray(distance_m) ** 2 * inv_two_effective_radius(refraction, k)
