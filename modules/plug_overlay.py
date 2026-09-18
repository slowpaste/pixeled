import math
import time

import numpy as np
from PIL import Image

from utils.plug_physics import Breach, Ripple
from utils.plug_watch import JackWatch, PowerWatch
from utils.ticker_line import TickerLine
from utils.tiny_font import TINY_HEIGHT


class PlugOverlay:
    """The laptop's sockets, felt through the panel.

    Two of them are level with the panel, on the right edge, and everything
    here starts from where they are:

    - the headphone jack, JACK_FROM_BOTTOM pixels up from the bottom corner,
      counting that corner as the first. A plug is about a pixel of the edge.
    - the USB-C port the charger goes into, across the middle of the battery
      gauge: the two rows at its centre, and half of the row above them.

    Plugging either in drops a ripple into the panel where it is (see Ripple):
    rings race out across whatever is showing, shoving it out and back in as
    they pass. Both share one set of settings, so tuning the ripple tunes it
    for both, while each starts from its own socket.

    Pulling either out breaches the panel there (see Breach): everything on it
    is sucked out through that socket, fast - the wider the socket, the wider
    the hole it goes through - and the panel is left empty for a moment. Then
    what it would be showing comes back the way it normally arrives:

    - a layout with a title row - the ticker over the gauges, the artwork's
      title over its cover, the overview's over its stack - rises from below
      under the row as the overlays do, while the row's line starts again
      from its right edge, braking to rest or running on at reading speed as
      it would after anything that interrupts it. The row is restarted
      through the same handover the overlays use, so the report being read
      picks up at the word it had got to.
    - anything else - the visualizer over the whole panel, or a layout with
      no ticker - rises from below whole.

    The overview's title has no handover to restart it through, so its row
    slides in from the right instead, ramping up and braking to rest as the
    line itself would.

    Not a module: it goes over whatever main.py composes, and is inert until
    a socket does something. The panel under it goes on being drawn while it
    is empty, so the gauges, the artwork's cycle and the tickers keep time and
    come back current rather than from where they were left.
    """

    # The headphone jack: its pixel row, counting the bottom corner as 1, and
    # how much of the edge a plug in it takes.
    JACK_FROM_BOTTOM = 8
    JACK_OPENING = 1.0
    # The power port, as rows of the panel: the middle two rows of the battery
    # gauge, which has rows 13 to 18, and half of the row above them.
    POWER_ROWS = (14.5, 17.0)
    TITLE_ROWS = TINY_HEIGHT    # rows 0-4, where the tickers scroll
    EMPTY_TIME = 0.2            # seconds the panel stays empty after draining
    RISE_TIME = 0.25            # seconds to rise back, as the artwork's cover does
    # px/s, overridden at runtime by pixeled-speed along with the tickers'.
    TEXT_SPEED = 24.0

    def __init__(self, width, height, jack_watch=None, power_watch=None):
        self.width = width
        self.height = height
        low, high = self.POWER_ROWS
        self.jack = _Port(width, height, height - self.JACK_FROM_BOTTOM + 0.5,
                          self.JACK_OPENING, jack_watch or JackWatch())
        self.power = _Port(width, height, (low + high) / 2, high - low,
                           power_watch or PowerWatch())
        self._ports = (self.jack, self.power)
        self._line = TickerLine(width, self.TEXT_SPEED)   # for its ramps
        self._kind = 'whole'
        self._state = None      # None, 'breach', 'empty' or 'return'
        self._draining = None   # the port it is going out of, while it is
        self._since = 0.0       # seconds in the state
        self._slide_title = False   # whether the title row is slid in here
        self._last = None       # the last frame shown, to breach with
        self._clock = None

    def set_scroll_speed(self, px_per_second):
        """The tickers' reading speed, which the title row ramps by."""
        self._line.text_speed = float(px_per_second)

    # ── Coming back ──────────────────────────────────────────────────────────

    def _start_return(self, layout, dt):
        kind, ticker = layout()
        self._state, self._since = 'return', 0.0
        self._kind = kind
        self._slide_title = kind == 'split' and ticker is None
        if kind == 'split' and ticker is not None:
            # Take the line and give it straight back empty, with nothing on
            # the panel and at rest: whatever it had next comes in from the
            # right edge from a standstill.
            ticker.hand_over(self.width)
            ticker.take_back(pieces=[], x=self.width, width=self.width,
                             speed=0.0, accel=self._line.accel(), dt=dt)

    def _title_offset(self):
        """Columns short of home the title row is, sliding in: from rest at
        the line's acceleration, braking at the same rate to rest in place."""
        accel = self._line.accel()
        total = 2 * math.sqrt(self.width / accel)
        t = min(self._since, total)
        if t < total / 2:
            gone = accel * t * t / 2
        else:
            gone = self.width - accel * (total - t) ** 2 / 2
        return round(self.width - gone), t >= total

    def _returning(self, frame):
        """`frame` as far back into place as it has come."""
        out = np.zeros_like(frame)
        rise = min(self._since / self.RISE_TIME, 1.0)
        top = self.TITLE_ROWS if self._kind == 'split' else 0
        shift = round((self.height - top) * (1.0 - rise))
        if self.height - shift > top:
            out[top + shift:] = frame[top:self.height - shift]
        home = rise >= 1.0
        if top:
            if self._slide_title:
                offset, arrived = self._title_offset()
                home = home and arrived
                if offset < self.width:
                    out[:top, offset:] = frame[:top, :self.width - offset]
            else:
                out[:top] = frame[:top]
        if home:
            self._state = None
        return out

    # ── Frame ────────────────────────────────────────────────────────────────

    def render(self, under, layout):
        """The panel's frame, with whatever the sockets are doing to it.

        `under` renders the frame as it would be without this, as a greyscale
        image. `layout` says how the frame `under` last drew is laid out:
        ('split', ticker) for one with a title row over the rest, where ticker
        is what has the row if it can be handed over, or None; or ('whole',
        None).
        """
        now = time.monotonic()
        dt = 0.0 if self._clock is None else min(max(now - self._clock, 0.0), 0.1)
        self._clock = now
        self._since += dt

        for port in self._ports:
            for event in port.watch.events():
                if event == 'out':
                    snapshot = self._last
                    if snapshot is None:
                        snapshot = np.asarray(under(), dtype=np.uint8)
                    for other in self._ports:
                        other.ripple.stop()
                    # Whichever socket was last pulled has the panel: a second
                    # one pulled mid-drain takes what is left out through its
                    # own hole rather than waiting its turn.
                    port.breach.start(snapshot)
                    self._state, self._since, self._draining = 'breach', 0.0, port
                elif event == 'in':
                    if self._state in ('breach', 'empty'):
                        self._start_return(layout, dt)
                    port.ripple.strike()
        if self._state == 'empty' and self._since >= self.EMPTY_TIME:
            # Before the panel under is drawn, so a title row restarted for
            # the return is drawn restarted from its very first frame.
            self._start_return(layout, dt)

        image = under()
        rippling = [port.ripple for port in self._ports if port.ripple.active]
        if self._state is None and not rippling:
            self._last = np.asarray(image, dtype=np.uint8)
            return image
        frame = np.asarray(image, dtype=np.uint8)
        if self._state == 'breach':
            breach = self._draining.breach
            breach.advance(dt)
            frame = breach.pixels(dt)
            if breach.drained:
                self._state, self._since, self._draining = 'empty', 0.0, None
        elif self._state == 'empty':
            frame = np.zeros_like(frame)
        elif self._state == 'return':
            frame = self._returning(frame)
        if self._state in (None, 'return'):
            # Two ripples at once are two lots of water in the one pool: the
            # second moves what the first has already moved.
            for ripple in rippling:
                ripple.advance(dt)
                frame = ripple.apply(frame)
        self._last = frame
        return Image.fromarray(frame, 'L')


class _Port:
    """One socket: where it is on the panel, and what it does to it."""

    def __init__(self, width, height, row, opening, watch):
        self.watch = watch
        self.ripple = Ripple(width, height, width, row)
        self.breach = Breach(width, height, width, row, opening)
