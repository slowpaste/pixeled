import math

import numpy as np

from utils.sound_medium import SoundMedium

GAMMA = SoundMedium.GAMMA       # the panel is linear; the eye is not

# Brightnesses as the eye sees them, 0 to 1.
FILL = 0.85                     # the wedge up to the volume
TRACK = 0.35                    # the rest of it
TRACK_COVER = 0.5               # how much of what is under the rest it hides
MUTED_FILL = 0.6                # the wedge up to the volume, muted
SLASH = 1.0
OVER = 1.0                      # the tallest column, fully boosted
OVER_RANGE = 0.5                # volume past 100% that takes it all the way


def _stepped(t, n, m):
    """Where along an axis m long a line is at t of n, to the nearest pixel,
    a tie going toward the nearer end, so the line is symmetric end to end."""
    v = t * m / n
    return math.ceil(v - 0.5) if 2 * t < n else math.floor(v + 0.5)


def _slash(rows, width):
    """The pixels of a line from the top-left corner to the bottom-right, one
    to each step along whichever way it is longer."""
    if rows == 1 or width == 1:
        return [(r, c) for r in range(rows) for c in range(width)]
    if width >= rows:
        n = width - 1
        return [(_stepped(c, n, rows - 1), c) for c in range(width)]
    n = rows - 1
    return [(r, _stepped(r, n, width - 1)) for r in range(rows)]


def volume_wedge(rows, width, level, muted):
    """The volume as a wedge rising to the right, filled from the left as far
    as `level` (1 is 100%), with a slash through it when muted.

    Returns (light, cover), each rows x width: light as the eye sees it, 0 to
    1, and how much of whatever is under the wedge it hides, 0 to 1. The fill
    and the slash hide it all; the rest of the wedge only TRACK_COVER of it,
    so what moves under it shows faintly through.

    The column the fill ends in is lit by the share of it filled, mixed as
    light rather than as the eye sees it, so the level moves smoothly through
    a column rather than a whole one at a time. Past 100% the tallest column
    brightens. Where the slash crosses the fill it is cut dark, so the level
    to unmute to still shows.
    """
    light = np.zeros((rows, width))
    cover = np.zeros((rows, width))
    in_fill = np.zeros((rows, width), dtype=bool)
    fill = MUTED_FILL if muted else FILL
    filled = min(max(level, 0.0), 1.0) * width
    over = min(max(level - 1.0, 0.0) / OVER_RANGE, 1.0)
    for c in range(width):
        top = rows - max(1, round((c + 1) * rows / width))
        share = min(max(filled - c, 0.0), 1.0)
        lit = fill
        if c == width - 1 and not muted:
            lit += (OVER - fill) * over
        mixed = share * lit ** GAMMA + (1 - share) * TRACK ** GAMMA
        light[top:, c] = mixed ** (1 / GAMMA)
        cover[top:, c] = share + (1 - share) * TRACK_COVER
        in_fill[top:, c] = share >= 0.5
    if muted:
        for r, c in _slash(rows, width):
            light[r, c] = 0.0 if in_fill[r, c] else SLASH
            cover[r, c] = 1.0
    return light, cover
