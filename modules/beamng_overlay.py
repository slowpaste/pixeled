import json
import logging
import os
import time

from PIL import Image

from modules.beamng_dash_module import BeamngDashModule
from utils.tiny_font import TINY_HEIGHT
from utils.ticker_line import TickerLine


class _DashTop:
    """The dashboard's top rows, as something that can travel along the title
    row the way the sound visualizer does.

    The line carries runs - words, or a picture with a width that knows how to
    draw itself - so the tachometer scrolls in where the next word would have
    gone, and comes to rest filling the row. That is the whole of what it takes
    for a full-panel layout to arrive the way everything else on this panel
    arrives, instead of appearing between two frames.
    """

    def __init__(self, overlay):
        self._overlay = overlay

    @property
    def width(self):
        return self._overlay.width

    def draw(self, image, x, y):
        top = self._overlay.top_rows()
        if top is not None:
            image.paste(top, (int(round(x)), y))


class BeamngOverlay:
    """The driving dashboard, over everything but the overview.

    Priority, top down: the overview covers this, this covers the artwork and
    its visualizer, and the artwork covers the gauges. A dashboard you are
    driving by should not be interrupted by the cover of whatever is playing,
    and the overview is the one thing that should interrupt anything.

    It is up only while BeamNG has the keyboard, not merely while it is
    running. A game left running behind a browser is not being driven, and the
    panel is better off showing what the desktop is doing; alt-tab back and the
    dash comes up again. Focus is the GNOME extension's to report - the service
    is a system unit with no session bus and cannot see a window - and with the
    extension gone, or the file never written, nothing here ever shows, which
    is the harmless way round.

    It arrives the way the cover does: the body rises from the bottom edge over
    what it is covering, and the top rows take the title row from whoever had
    it at a word boundary, so words already on the panel finish going past
    instead of being cut in half. Leaving is the same in reverse, the row
    handed back as the body starts down.

    The dashboard itself is BeamngDashModule, built from the entry in
    config.json.beamng, which is why that file is still the place its scales
    and its feel are tuned. It used to be swapped into config.json wholesale
    by a monitor thread watching for the process; a layout swap cannot be
    layered over anything, cannot slide, and left the panel on the dashboard
    layout whenever the game was merely running.
    """

    DEFAULT_STATE_FILE = os.path.expanduser('~/.cache/pixeled/focus')
    DEFAULT_CONFIG = 'config.json.beamng'
    POLL_INTERVAL = 0.1     # seconds; matches the workspace indicator
    TITLE_ROWS = TINY_HEIGHT
    # Seconds for the body to rise or fall, timed like the artwork's cover so
    # the two read as the same panel doing the same thing.
    SLIDE_TIME = 0.25
    MAX_DT = 0.25

    def __init__(self, width, height, app_match='beamng', state_file=None,
                 config_file=None, scroll_speed=24.0):
        """
        :param app_match: matched against the focused window's WM class, its
            instance and its title, lowercased, as a substring. A class of
            BeamNG.drive.x64 and a title of BeamNG.drive both carry it, and
            nothing else on a desktop does.
        :param state_file: where the GNOME extension reports the focus.
        :param config_file: the layout the dashboard's settings are read from.
        """
        self.width = width
        self.height = height
        self.app_match = app_match.lower()
        self._state_file = state_file or self.DEFAULT_STATE_FILE
        self._config_file = config_file or self.DEFAULT_CONFIG

        self._dash = BeamngDashModule(height=height, **self._settings())
        self._top = _DashTop(self)
        self._line = TickerLine(width, float(scroll_speed))

        self._focused = False
        self._checked_at = 0.0
        self._stamp = None
        self._clock = None
        self._slide_pos = 0.0   # 0 clear of the bottom edge, 1 home
        self._owns_line = False
        self._top_at = None
        self._ticker = None     # what had the row under this, for handing back
        self._frame = None      # the dashboard this frame, for _DashTop
        self._parting = None    # the last one, to slide back out with

    # ── settings ─────────────────────────────────────────────────────────────
    def _settings(self):
        """The dashboard's own arguments out of config.json.beamng.

        Anything the module does not take is left out rather than passed on: a
        layout file carries position and height for the compositor's sake, and
        a stale key in it should not take the panel down.
        """
        try:
            with open(self._config_file) as f:
                entries = json.load(f)
        except (OSError, ValueError) as e:
            logging.warning("BeamngOverlay: no settings from %s (%s)",
                            self._config_file, e)
            return {}
        wanted = BeamngDashModule.__init__.__code__.co_varnames
        for entry in entries:
            if entry.get('module') == 'BeamngDashModule':
                return {k: v for k, v in entry.items()
                        if k in wanted and k not in ('self', 'height')}
        return {}

    def set_scroll_speed(self, px_per_second):
        self._line.text_speed = float(px_per_second)

    # ── focus ────────────────────────────────────────────────────────────────
    def _read_focus(self):
        """Whether the game has the keyboard, re-read at most every poll.

        Unreadable, missing or empty all mean no: the panel's own layouts are
        what it shows when nothing says otherwise.
        """
        now = time.monotonic()
        if now - self._checked_at < self.POLL_INTERVAL:
            return self._focused
        self._checked_at = now
        try:
            stamp = os.stat(self._state_file).st_mtime_ns
        except OSError:
            self._focused = False
            self._stamp = None
            return False
        if stamp == self._stamp:
            return self._focused
        self._stamp = stamp
        try:
            with open(self._state_file) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return self._focused
        was = self._focused
        self._focused = any(self.app_match in str(data.get(key, '')).lower()
                            for key in ('app', 'instance', 'title'))
        if self._focused != was:
            logging.info("BeamngOverlay: %s focused",
                         data.get('app') or 'nothing' if not self._focused
                         else data.get('app'))
        return self._focused

    def showing(self):
        """Whether the dash has the panel, or is on its way to having it."""
        return self._focused or self._slide_pos > 0.0

    def full_panel(self):
        """Settled, with the title row its own and the top rows come to rest on
        it: nothing under this shows, so the sockets treat the panel as whole.

        Mid-hand-over it is not: the row still has words going past on it, and
        they belong to the layer this took it from.
        """
        return (self._owns_line and self._slide_pos >= 1.0
                and self._top_at is not None and self._line.speed == 0.0
                and self._line.x(self._top_at) == 0)

    # ── the title row ────────────────────────────────────────────────────────
    def _own_line(self, ticker):
        """Take the row, from `ticker` at a word boundary if there is one."""
        if ticker is not None:
            self._line.take(ticker)
        else:
            self._line.reset()
        self._owns_line = True
        self._top_at = self._line.rest(self._top)

    def _let_go(self, under, dt):
        """Give the row back to whatever is under this, and return that."""
        if not self._owns_line:
            return under() if under is not None else None
        handback, _ = self._line.give(dt)
        self._owns_line = False
        self._top_at = None
        words = dict(handback, width=self.width, dt=dt)
        return under(words) if under is not None else None

    def ticker(self):
        """What has the title row while this is on the panel, for the
        overview to share it with."""
        return self if self._owns_line else self._ticker

    def hand_over(self, width):
        """Give up the line at the next word, for the overview's title."""
        handback, _ = self._line.give(0.0)
        self._owns_line = False
        return handback['pieces'], handback['x'], handback['speed']

    def take_back(self, pieces, x, width, speed, accel, dt):
        """Resume after the overview's words, the dashboard following them."""
        self._line.take_back(pieces, x, speed, dt)
        self._owns_line = True
        self._clock = time.monotonic() - dt
        self._top_at = self._line.rest(self._top)

    # ── drawing ──────────────────────────────────────────────────────────────
    def top_rows(self):
        """The dashboard's top rows, for _DashTop to put on the line."""
        frame = self._frame or self._parting
        if frame is None:
            return None
        return frame.crop((0, 0, self.width, self.TITLE_ROWS))

    def _move_slide(self, direction, dt):
        step = dt / self.SLIDE_TIME if self.SLIDE_TIME > 0 else 1.0
        self._slide_pos = min(1.0, max(0.0, self._slide_pos + direction * step))

    def _shift(self):
        """Rows the body is still short of home, from the bottom edge."""
        travel = self.height - self.TITLE_ROWS
        return int(round((1.0 - self._slide_pos) * travel))

    def _compose(self, frame, shift, backdrop, dt):
        """The dashboard over `backdrop`, its body `shift` rows down the panel
        and its top rows on the title row once the row is this one's."""
        if backdrop is None:
            image = frame.copy()
        else:
            image = backdrop[0].copy()
            body = frame.crop((0, self.TITLE_ROWS, self.width, self.height - shift))
            image.paste(body, (0, self.TITLE_ROWS + shift))
        if self._owns_line and self._top_at is not None:
            # The goal is the scroll that brings the top rows to rest at the
            # left edge, wanted at a standstill: the same ramp the visualizer
            # comes in on.
            self._line.move(dt, self._top_at + self.width, 0.0)
            top = Image.new('L', (self.width, self.TITLE_ROWS), 0)
            self._line.draw(top, [(self._top, self._top_at)])
            image.paste(top, (0, 0))
        return image

    def render(self, under=None):
        """A full-panel frame, or None to let what is under it through."""
        now = time.monotonic()
        dt = 0.0 if self._clock is None else min(max(now - self._clock, 0.0), self.MAX_DT)
        self._clock = now

        focused = self._read_focus()

        if not focused:
            if self._slide_pos <= 0.0:
                if self._owns_line:
                    self._let_go(under, dt)
                self._frame = self._parting = None
                self._ticker = None
                return None
            # The row goes back before anything under draws, so its ticker
            # carries on from it this very frame.
            backdrop = self._let_go(under, dt)
            self._move_slide(-1, dt)
            if self._slide_pos <= 0.0 or self._parting is None:
                self._frame = self._parting = None
                return None
            self._ticker = backdrop[2] if backdrop else None
            return self._compose(self._parting, self._shift(), backdrop, dt)

        self._frame = self._dash.render(self.width)
        self._parting = self._frame
        self._move_slide(+1, dt)
        shift = self._shift()
        backdrop = None
        if shift > 0:
            backdrop = under() if under is not None else None
            self._ticker = backdrop[2] if backdrop else None
            if not self._owns_line and self._ticker is None:
                # Nothing to share the row with: it rises with the body.
                self._own_line(None)
        elif not self._owns_line:
            # Home, under the title row: the row becomes this one's, from
            # where it was drawn last frame.
            self._own_line(self._ticker)
        return self._compose(self._frame, shift, backdrop, dt)
