import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from collections import namedtuple
from PIL import Image, ImageFilter, ImageOps

from utils.sound_medium import SoundMedium
from utils.sound_visualizer import sound_visualizer
from utils.ticker_line import TickerLine
from utils.tiny_font import (TINY_ADVANCE, TINY_HEIGHT, draw_tiny_text,
                             sanitize_tiny_text, tiny_text_width)

# What the shell is publishing right now. `label` is the words to scroll - a
# tuple of the title and the artist, whichever are known - or None for a
# player that says nothing about what it is playing. `ends_at` is the wall
# clock time the track will end, or None where the player does not say.
State = namedtuple('State', 'path label ends_at')


class ArtworkOverlay:
    """What is playing: its cover, and the sound of it along the top.

    Not a module: the compositor packs modules into a column of the panel, and
    this takes all 34 rows for as long as it has something to show, so main.py
    asks it for a frame before falling back to the composed one.

    Laid out like the overview. The top five rows are the ticker's, and show
    the sound visualizer, which scrolls in and stays. Under a blank row, the
    other 28 are the cover, drifting back and forth for as long as something
    is playing. Covers survive this well - big shapes, high contrast - and the
    art changes by itself every few minutes, which is where the variety comes
    from.

    Once a new song has opened on the visualizer (see below), it goes round a
    cycle:

    1. The song's name. Whatever was showing scrolls up off the top, and the
       title and then the artist follow it up the panel in the ticker's
       letters - turned a quarter clockwise, so they read top to bottom with
       the head tilted right, the title along the right edge and the artist
       along the left - with the cover coming up behind them.
    2. The cover, panning across and back: two pans.
    3. The visualizer, the whole panel. It grows down out of its row, pushing
       the cover off the bottom, and stays for FULL_TIME before shrinking back
       up into the row, and the cycle comes round to the name again.

    A track's last OUTRO_TIME goes to the visualizer too, wherever the cycle
    is, where the player says how long the track is and how far it has got,
    and it stays out through the change of song into the next one's opening.

    A new song opens on the visualizer instead: it grows over whatever the
    cycle was showing, or stays out if it already was, for INTRO_TIME, so the
    song's first moments are the sound alone, and only then does the cycle
    start from the song's name. The visualizer only grows from rest in its
    place at the left of the row, so it never grows out of words still going
    past.

    Sideways rather than cropping because the panel is 9 wide: scaling a cover
    to fit that throws away most of it, while scaling to the height and
    panning shows the whole width eventually. A square cover becomes 28px.

    It slides rather than cutting, both ways: the cover rises from below over
    the gauges it is about to cover when playback starts, and drops back down
    off the bottom edge to uncover them when it stops. A cut says only that the
    panel changed; a direction says which of the two just happened.

    Only the cover slides. The ticker stays where it is and keeps scrolling
    until the cover arrives under it, and then the line changes hands at a
    word boundary the way it does for the overview: the words already on the
    panel carry on past, and the visualizer comes in where the next word would
    have, the words ahead of it hurried off - or stays put, if it was what the
    ticker was showing. Stopping gives the line back to the ticker as the
    cover starts down. With no ticker under it, the row rises and falls glued
    to the cover instead.

    The overview slides over this as it does over the gauges, and treats this
    as the ticker: its title takes the line from the visualizer, and gives it
    back when it closes - to this if something is still playing, otherwise to
    the gauges' ticker, as the cover leaves.

    Leaving means outliving the state that justified it - the shell stops
    publishing the art the moment playback stops, so the cover has to be
    remembered for the frames it takes to get off the panel. It is the same
    travel in reverse through the same positions, so playback resumed part way
    out picks up from where the cover had got to instead of snapping.

    The pan moves in fractions of a pixel. At ~60fps a 4.5px/s pan advances
    under a tenth of a column a frame, and stepping only on whole columns would
    turn that into a visible tick every fifth of a second; blending the two
    columns either side of the true position carries the fraction in
    brightness instead, so the picture drifts.
    """

    DEFAULT_STATE_FILE = os.path.expanduser('~/.cache/pixeled/artwork')
    CACHE_DIR = os.path.expanduser('~/.cache/pixeled/artwork-remote')
    POLL_INTERVAL = 0.1     # seconds; matches the workspace indicator
    TITLE_ROWS = TINY_HEIGHT        # rows 0-4, where the transit ticker scrolls
    ART_TOP = TITLE_ROWS + 1        # one blank row between title and cover
    # px/s, ping-ponging across the cover. Slow, because the picture is there to
    # be looked at rather than read, and still under a pixel a frame on stock
    # firmware's ~6fps should the panel ever be back on it.
    MUSIC_PAN_SPEED = 4.5
    MAX_STRIP = 120         # px; a very wide image would otherwise pan for minutes
    # Preparing a cover for 9x28 LEDs, tuned for telling covers apart rather
    # than for faithful brightness; see _strip.
    CONTRAST_CUTOFF = 2     # percent of darkest and brightest pixels ignored
    SHARPEN = ImageFilter.UnsharpMask(radius=1.0, percent=70, threshold=0)
    DISPLAY_GAMMA = 1.8
    CACHE_ENTRIES = 6       # decoded strips kept, keyed by path and mtime
    # Seconds to rise over the gauges, or drop off them. Timed like the
    # overview's slide rather than counted in frames: this used to be two
    # refreshes, which at the old greyscale rate of ~5.9fps was one halfway
    # frame and ~340ms of travel, and at ~60fps would be a 33ms cut.
    SLIDE_TIME = 0.25
    # px/s, overridden at runtime by pixeled-speed along with the ticker's.
    TEXT_SPEED = 24.0
    # Rows along the panel between the title and the artist. They go past on
    # opposite edges, which is what tells them apart, but their letters
    # share the middle columns, so one has to be gone before the next comes.
    PART_GAP = TINY_ADVANCE
    PART_MARGIN = 1         # columns between each and its edge
    PANS_BEFORE_FULL = 2    # pans of the cover before the visualizer grows
    # Seconds the visualizer has the whole panel. Long enough to stop glancing
    # at it and just watch - about a verse and a chorus - and short enough that
    # the name and the cover come round again before they feel overdue.
    FULL_TIME = 55.0
    # Seconds a new song opens on the whole-panel visualizer before its name.
    # Long enough to take in how it starts; short enough that the name still
    # arrives while the song is new.
    INTRO_TIME = 10.0
    # Seconds before a track ends that the visualizer takes the panel for the
    # rest of it - through a fade-out, and on into the next song's opening.
    OUTRO_TIME = 20.0
    # Seconds past its end a track is still taken to be ending: players take
    # a moment to move on, and a track that is still going well past its
    # length was given a wrong one, so its ending is not to be believed.
    OUTRO_GRACE = 5.0
    # Seconds a track has to go on not ending before the ending is let go of.
    # Changing track, Cider says the position went back to the start a tenth
    # of a second before it says the track changed, which for that moment
    # reads as the old track seeked back from its ending: the cycle moved on
    # to the old track's name, and the new one's opening took the panel back.
    OUTRO_HOLD = 1.0
    # Seconds the cover stays through playback seeming to stop. A player can
    # blink out of Playing between tracks or while it seeks, and the cover
    # slid off the panel and straight back in, dropping the grown visualizer
    # into its row on the way and growing it out again.
    LEAVE_HOLD = 0.5
    GROW_TIME = 0.9         # seconds to grow over the panel, or shrink back
    NARROW_PAN_TIME = 6.0   # seconds that count as a pan, for a cover with no room
    SIDE_GAP = 3            # rows between the cover and the title going past it
    SIDE_RAMP_TIME = 0.25   # seconds from rest to reading speed, and back

    def __init__(self, width, height, state_file=None):
        self.width = width
        self.height = height
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self._next_poll = 0.0
        self._state = None
        self._strips = {}
        self._key = None        # the cover the current pan belongs to
        self._pan = 0.0
        self._direction = 1
        self._panned_at = 0.0
        self._fetching = set()
        self._slide_pos = 0.0   # 0 is clear of the bottom edge, 1 home
        self._slide_at = None   # when the slide last moved; None seeds the clock
        self._parting = None    # last frame shown, kept to slide back out
        self._clock = None      # when the title last moved; None seeds the clock
        self._ticker = None     # the gauges' ticker, last slide frame
        self._line = TickerLine(width, self.TEXT_SPEED)
        self._owns_line = False  # whether the title row is this one's to draw
        self._visualizer = sound_visualizer(width)
        self._vis_at = None     # u of the visualizer on the line
        self._pans = 0          # pans of this cover since the title last went by
        self._still = 0.0       # seconds a cover with no room has sat, toward a pan
        self._side = None       # the title going up the panel, while it does
        self._phase = 'art'     # 'side', 'art' or 'full': where the cycle is
        self._song = None       # the label the cycle last started for
        self._grow = 0.0        # 0 the visualizer in its row, 1 the whole panel
        self._full_left = 0.0   # seconds of the whole panel still to come
        self._last_area = None  # what the cover area showed last frame
        self._pending_side = None  # (old, label) for the name, once it can start
        self._full_area = None  # what the cover area showed as the visualizer grew
        self._outro = False     # whether the track's ending has the panel
        # Set from outside to keep the visualizer over the whole panel for as
        # long as it is, whatever the cycle would do - while it is being tuned.
        self.hold_full = False
        self._unending_at = None  # when the ending last stopped being believed
        self._full_before = 0.0   # whole-panel time left when the ending took it
        self._held = None       # (state, strip, time) last drawn from, to hold

    def set_scroll_speed(self, px_per_second):
        """Retune the title's scroll rate while running."""
        self._line.text_speed = float(px_per_second)

    @property
    def _text_speed(self):
        return self._line.text_speed

    # ── State ────────────────────────────────────────────────────────────────

    def _read_state(self):
        """Latest published artwork state, or None.

        Unlike the workspace indicator, a bad or missing read clears rather
        than holding the last value: this covers the gauges, so anything that
        stops publishing - a shell restart, the extension being disabled -
        has to give the panel back rather than freeze a picture over it.
        """
        now = time.monotonic()
        if now < self._next_poll:
            return self._state
        self._next_poll = now + self.POLL_INTERVAL

        try:
            with open(self.state_file) as f:
                data = json.load(f)
        except (OSError, ValueError):
            self._state = None
            return None

        if data.get('playing') and data.get('art'):
            self._state = State(data['art'], self._label_of(data),
                                self._ends_at_of(data))
        else:
            self._state = None
        return self._state

    @staticmethod
    def _ends_at_of(data):
        try:
            ends_at = float(data.get('ends_at'))
        except (TypeError, ValueError):
            return None
        return ends_at if math.isfinite(ends_at) else None

    def _label_of(self, data):
        """Title and artist, as far as they are known, in words this font can
        draw, or None where there is nothing worth saying."""
        parts = [sanitize_tiny_text(data.get(k) or '') for k in ('title', 'artist')]
        return tuple(p for p in parts if p) or None

    def full_panel(self):
        """Whether the visualizer has the whole panel, grown and home."""
        return self._owns_line and self._grow >= 1.0 and self._slide_pos >= 1.0

    def playing(self):
        """Whether there is a cover to show, and so a title to own the row.

        Asked before this renders, by whoever has words to give back, so they
        go to the track rather than to the gauges' ticker under it.
        """
        return self._current()[1] is not None

    def _current(self):
        """The state to draw and its strip, or (state, None) for nothing.

        Held over what the shell publishes in two cases, so neither sends the
        cover off the panel only to bring it straight back. A new cover still
        downloading keeps the last one showing until it lands - the new song
        opens on the visualizer anyway, over whatever was there. And playback
        seeming to stop keeps the last state for LEAVE_HOLD first, for the
        players that blink out of Playing between tracks and while seeking.
        Neither holds once the cover has left: there is nothing to keep.
        """
        state = self._read_state()
        strip = self._strip(state.path) if state is not None else None
        now = time.monotonic()
        held = self._held
        if strip is not None:
            self._held = (state, strip, now)
        elif held is not None:
            if state is not None and state.path in self._fetching:
                strip = held[1]
                self._held = (state, strip, now)
            elif now - held[2] < self.LEAVE_HOLD:
                state, strip = held[0], held[1]
        return state, strip

    # ── Images ───────────────────────────────────────────────────────────────

    def _local_path(self, path):
        """A file to open, fetching http(s) art in the background if need be.

        Returns None while a fetch is in flight, so a slow server costs the
        gauges nothing - the panel simply keeps drawing them until the art
        lands, and picks it up on a later frame.
        """
        if not path.startswith(('http://', 'https://')):
            return path

        cached = os.path.join(self.CACHE_DIR,
                              hashlib.sha1(path.encode()).hexdigest())
        if os.path.exists(cached):
            return cached
        if path not in self._fetching:
            self._fetching.add(path)
            threading.Thread(target=self._fetch, args=(path, cached),
                             daemon=True).start()
        return None

    def _fetch(self, url, target):
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read()
            partial = f'{target}.part'
            with open(partial, 'wb') as f:
                f.write(data)
            os.replace(partial, target)   # readers never see a partial file
        except Exception:
            pass                          # no art is a fallback, not an error
        finally:
            self._fetching.discard(url)

    def _strip(self, path):
        """The image scaled to the rows under the title, ready to slice a
        window from.

        Three adjustments after scaling, all aimed at a cover being
        recognisable at 28 rows rather than at its brightness being accurate:

        - Contrast is stretched to the full range, ignoring the extreme couple
          of percent at each end, so one white logo or a black border cannot
          stop the rest of the picture from using the range.
        - It is sharpened at panel resolution. Scaling a cover down to a few
          dozen pixels averages its edges away, and the edges are most of what
          makes a shape recognisable at this size.
        - The result is mapped through a display curve. Cover art is stored
          encoded for a monitor, where a mid grey value is meant to show much
          dimmer than half brightness, but the panel's brightness is linear in
          the value it is sent, so sent as-is a cover comes out washed out.
          Decoding with a monitor's usual 2.2 would be accurate; 1.8 stops
          short on purpose, because the firmware's default 20% brightness
          leaves only 52 real steps: at 2.2 the darkest 16% of a cover goes
          black and the rest of its darker third shares four steps, at 1.8
          only the darkest 11% goes black and that third gets seven.

        All of it is done here, once per image, which also keeps the result
        stable as the window travels - anything computed per slice would
        re-derive itself from whichever 9px happened to be showing and crawl.
        """
        local = self._local_path(path)
        if local is None:
            return None
        try:
            stamp = os.path.getmtime(local)
        except OSError:
            return None

        key = (local, stamp)
        strip = self._strips.get(key)
        if strip is not None:
            return strip

        try:
            rows = self.height - self.ART_TOP
            with Image.open(local) as image:
                image.draft('L', (self.MAX_STRIP * 4, rows * 4))
                image = image.convert('L')
                width = max(self.width,
                            min(round(rows * image.width / image.height),
                                self.MAX_STRIP))
                strip = ImageOps.autocontrast(
                    image.resize((width, rows), Image.LANCZOS),
                    cutoff=self.CONTRAST_CUTOFF)
                strip = strip.filter(self.SHARPEN).point(
                    [round(255 * (i / 255) ** self.DISPLAY_GAMMA) for i in range(256)])
        except (OSError, ValueError):
            return None

        if len(self._strips) >= self.CACHE_ENTRIES:
            self._strips.clear()
        self._strips[key] = strip
        return strip

    # ── Panning ──────────────────────────────────────────────────────────────

    def _column(self, travel):
        """Left edge of the window this instant, in fractional columns.

        The pan ping-pongs, because a cover has two edges and wrapping past one
        would show a seam that is not in the picture.
        """
        now = time.monotonic()
        # Capped, because this is not asked for a frame while the overview
        # covers it: an overview held open for a minute would otherwise spend
        # that whole minute in one step and fling the window at an edge.
        elapsed, self._panned_at = min(now - self._panned_at, 0.25), now
        if travel <= 0:
            # No wider than the panel; nowhere to pan, so time stands in for
            # the pans before the visualizer grows.
            self._still += elapsed
            if self._still >= self.NARROW_PAN_TIME:
                self._still -= self.NARROW_PAN_TIME
                self._pans += 1
            return 0.0

        self._pan += self._direction * self.MUSIC_PAN_SPEED * elapsed
        if self._pan >= travel:
            self._pans += self._direction > 0     # a pan ends on turning round,
            self._pan, self._direction = float(travel), -1
        elif self._pan <= 0:
            self._pans += self._direction < 0     # not on sitting at the start
            self._pan, self._direction = 0.0, 1
        return self._pan

    def _window(self, strip, x):
        """The panel-wide slice of the strip at fractional column x.

        A blend of the whole-column windows either side, weighted by how far
        between them x is. Blending neighbouring columns is exactly what a
        sub-pixel shift of the picture would sample, so the motion is smooth
        without resampling the strip every frame.
        """
        left = int(x)
        frame = strip.crop((left, 0, left + self.width, strip.height))
        fraction = x - left
        if fraction > 0.0 and left + self.width < strip.width:
            right = strip.crop((left + 1, 0, left + 1 + self.width, strip.height))
            frame = Image.blend(frame, right, fraction)
        return frame

    # ── Title ────────────────────────────────────────────────────────────────

    # The title row is a TickerLine, as the overview's is: words carried over
    # from whoever had the row, and then the sound visualizer, at rest.

    def _own_line(self, ticker):
        """Take the row: from `ticker` at a word boundary, or with no ticker
        to take it from, fresh, the visualizer entering from the right edge."""
        if ticker is not None:
            self._line.take(ticker)
        else:
            self._line.reset()
        self._owns_line = True
        self._vis_at = self._line.rest(self._visualizer)

    def ticker(self):
        """What has the title row while this is on the panel, for the
        overview to share it with."""
        return self if self._owns_line else self._ticker

    def hand_over(self, width):
        """Give up the line at the next word, for the overview's title.

        The visualizer stays as grown as it was, under the overview, for the
        panel to be the same when the overview closes as when it opened. The
        overview's row draws the top of it while it is still on the line.
        """
        handback, _ = self._line.give(0.0)
        self._owns_line = False
        return handback['pieces'], handback['x'], handback['speed']

    def take_back(self, pieces, x, width, speed, accel, dt):
        """Resume after the overview's words, the visualizer following them.

        `dt` is the overview's last frame time, standing in for the one this
        missed, since the line's own clock has been stopped under it.
        """
        self._line.take_back(pieces, x, speed, dt)
        self._owns_line = True
        self._clock = time.monotonic() - dt
        self._vis_at = self._line.rest(self._visualizer)

    def _draw_title(self, image, dt):
        """Move the line on toward the visualizer at rest, and draw it."""
        line = self._line
        line.move(dt, self._vis_at + self.width, 0.0)
        line.draw(image, [(self._visualizer, self._vis_at)])

    # ── Title, sideways ──────────────────────────────────────────────────────

    # Worked out lying down, as a line as long as the cover is tall: the cover
    # turned a quarter anticlockwise, a gap, the title, the artist, a gap, the
    # cover again, all scrolling left. Stood up a quarter clockwise, left is
    # up and the top edge is the right, and the cover and the words both read
    # the right way. The title runs along the top - the panel's right edge -
    # and the artist after it along the bottom, its left; either alone runs
    # down the middle.

    def _start_side(self, old, cover, label):
        """Scroll `old` - what the cover area shows now, or None for nothing -
        up and away, the label after it, and `cover` in to rest behind."""
        rows = self.height - self.ART_TOP
        at = rows + (self.SIDE_GAP if old is not None else 0)
        if len(label) == 1:
            edges = [(self.width - TINY_HEIGHT) // 2]
        else:
            edges = [self.PART_MARGIN, self.width - self.PART_MARGIN - TINY_HEIGHT]
        placed = []
        for text, y in zip(label, edges):
            placed.append((text, at, y))
            at += tiny_text_width(text) + self.PART_GAP
        at -= self.PART_GAP
        self._side = dict(
            placed=placed, scroll=0.0, speed=0.0,
            old=old.transpose(Image.ROTATE_90) if old is not None else None,
            cover=cover.transpose(Image.ROTATE_90),
            total=at + self.SIDE_GAP)

    def _side_frame(self, dt):
        """The cover area this instant, or None once the cover is back home.

        From rest up to reading speed, and braking to rest as the cover comes
        back into place, at the same rate the ticker ramps at.
        """
        side = self._side
        accel = self._text_speed / self.SIDE_RAMP_TIME
        left = side['total'] - side['scroll']
        speed = min(self._text_speed, side['speed'] + accel * dt,
                    math.sqrt(2 * accel * max(left, 0.0)))
        side['scroll'] += max(speed, 1.0) * dt
        side['speed'] = speed
        if side['scroll'] >= side['total']:
            self._side = None
            self._panned_at = time.monotonic()   # the pan carries on from here
            return None

        rows = self.height - self.ART_TOP
        lying = Image.new('L', (rows, self.width), 0)
        x = -math.floor(side['scroll'])
        if side['old'] is not None:
            lying.paste(side['old'], (x, 0))
        for text, at, y in side['placed']:
            draw_tiny_text(lying, text, x + at, y)
        lying.paste(side['cover'], (x + side['total'], 0))
        return lying.transpose(Image.ROTATE_270)

    # ── The cycle ────────────────────────────────────────────────────────────

    def _visualizer_at_rest(self):
        line = self._line
        return (self._owns_line and self._vis_at is not None and line.speed == 0.0
                and line.x(self._vis_at) == 0)

    def _visualizer_cells(self):
        """How tall the visualizer is drawn, in its world's cells: its row at
        rest, the whole panel fully grown, eased in between."""
        small = self.TITLE_ROWS * SoundMedium.SCALE
        full = self.height * SoundMedium.SCALE
        eased = self._grow * self._grow * (3 - 2 * self._grow)
        return round(small + (full - small) * eased)

    def _move_grow(self, dt):
        """Grow the visualizer toward the whole panel while the cycle is on
        it, and shrink it back otherwise.

        Only from rest in its place, so it never grows out of words still going
        along the row. Already grown, it stays so while words go past over its
        top - the overview's, handed back as it closes. Dropping back into the
        row at once is for when the row is not this view's at all: the cover
        arriving under the gauges' ticker, or leaving.

        It used to drop back for words on the row too, and closing the
        overview over the whole-panel visualizer showed the cover under it
        while the overview's words went by, before growing out over it again.
        """
        if not self._owns_line:
            self._grow = 0.0
            return
        step = dt / self.GROW_TIME
        if self._phase != 'full':
            self._grow = max(0.0, self._grow - step)
        elif self._visualizer_at_rest():
            self._grow = min(1.0, self._grow + step)

    def _follow_song(self, state):
        """Start the cycle over when the song changes."""
        if state.label == self._song:
            return
        self._song = state.label
        self._pans, self._still = 0, 0.0
        self._side = None
        self._outro = False     # the last song's ending, not this one's
        self._unending_at = None
        if state.label is None:
            self._phase = 'art'
            return
        # Grow over whatever the cover area shows, which stays where it was
        # under the visualizer; or, already grown, just stay.
        self._phase, self._full_left = 'full', self.INTRO_TIME
        self._full_area = self._last_area

    def _ending(self, state):
        if state.ends_at is None:
            return False
        left = state.ends_at - time.time()
        return -self.OUTRO_GRACE < left <= self.OUTRO_TIME

    def _follow_ending(self, state):
        """Give the visualizer the panel for the end of the track, and let the
        cycle go on if it turns out not to be ending after all - seeked back,
        say. For as long as it is ending, the whole panel does not run out.

        Only once it has gone OUTRO_HOLD not ending, though, since the start
        of the next track can first read as this one seeked back. And the
        whole panel gets back whatever time it had left when the ending took
        it: a new track can also, for a moment, read as ending - Cider gives
        the last one's position with the new one's length - and that cut the
        new song's opening short."""
        ending = self._ending(state)
        if ending:
            self._unending_at = None
        if ending and not self._outro:
            self._outro = True
            self._full_before = self._full_left if self._phase == 'full' else 0.0
            if self._phase != 'full':
                self._phase, self._full_area = 'full', self._last_area
                self._side = None
            self._full_left = math.inf
        elif self._outro and not ending:
            now = time.monotonic()
            if self._unending_at is None:
                self._unending_at = now
            if now - self._unending_at >= self.OUTRO_HOLD:
                self._outro = False
                self._unending_at = None
                if self._full_left == math.inf:
                    self._full_left = self._full_before

    def _cycle(self, state, strip, dt):
        """Move the cycle on; returns what the cover area shows, or None."""
        self._follow_ending(state)
        if self.hold_full and self._phase != 'full':
            # Grown over whatever is showing, as for a track's ending; let go,
            # the cycle carries on from the visualizer's time left, or none.
            self._phase, self._full_area = 'full', self._last_area
            self._full_left = 0.0
            self._side = None
        travel = strip.width - self.width
        if self._phase == 'side':
            if self._grow > 0.0:
                return None         # the visualizer is still shrinking back
            if self._side is None:
                old, label = self._pending_side
                self._start_side(old, self._window(strip, self._pan), label)
            area = self._side_frame(dt)
            if area is not None:
                return area
            self._phase, self._pans, self._still = 'art', 0, 0.0
        if self._phase == 'art':
            area = self._window(strip, self._column(travel))
            if self._pans >= self.PANS_BEFORE_FULL:
                self._phase, self._full_left = 'full', self.FULL_TIME
                self._full_area = area
            return area
        # The whole panel: what the cover area showed waits under it, for as
        # long as any of it is still showing below the visualizer.
        if self._grow >= 1.0 and not self.hold_full:
            self._full_left -= dt
            if self._full_left <= 0.0:
                self._pans, self._still = 0, 0.0
                if self._song is None:
                    # Nothing to say: straight back to the cover.
                    self._phase = 'art'
                    self._panned_at = time.monotonic()
                else:
                    self._phase = 'side'
                    self._pending_side = (None, self._song)
        return self._full_area

    # ── Arriving and leaving ─────────────────────────────────────────────────

    def _shift(self):
        """Rows the cover sits below home at this point in the slide.

        Arriving and leaving move the same position through the same rows, one
        up and one down, which is what lets a reopen part way out reverse
        instead of snapping - there is one slide, not two animations.
        """
        return round(self.height * (1.0 - self._slide_pos))

    def _move_slide(self, direction):
        """Advance the slide toward home (+1) or off the bottom edge (-1).

        dt is clamped like the overview's slide: this is not asked for a frame
        while the overview covers it, and the time spent under there should
        not be spent in one step when it is uncovered.
        """
        now = time.monotonic()
        dt = 0.0 if self._slide_at is None else min(max(now - self._slide_at, 0.0), 0.05)
        self._slide_at = now
        self._slide_pos = min(1.0, max(0.0, self._slide_pos + direction * dt / self.SLIDE_TIME))

    def _backdrop(self, under, handback=None):
        """What the cover slides over: the gauges, the (top, height) of rows
        in them to pull up with it, and their ticker. `handback` is the
        title's words for that ticker, given before the gauges draw."""
        base, pulled, ticker = (under(handback) if under is not None
                                else (None, (), None))
        if base is None or base.size != (self.width, self.height):
            base, pulled, ticker = Image.new('L', (self.width, self.height), 0), (), None
        self._ticker = ticker
        return base, pulled, ticker

    def _let_go(self, under, dt):
        """Give the line back to the gauges' ticker, and return the backdrop.

        With no ticker there to take it the title keeps the row, and leaves
        glued to the cover.
        """
        handback, _ = self._line.give(dt) if self._owns_line else (None, None)
        # The world the visualizer grew goes back into the row, for the ticker
        # to carry on drawing.
        self._visualizer.settle()
        backdrop = self._backdrop(under, handback)
        if self._owns_line and backdrop[2] is not None:
            self._owns_line = False
        return backdrop

    def _compose(self, area, dt):
        """The frame: the row, or the visualizer grown down from it, and the
        cover area under that, pushed down by as much as it has grown."""
        frame = Image.new('L', (self.width, self.height), 0)
        cells = self._visualizer_cells()
        small = self.TITLE_ROWS * SoundMedium.SCALE
        if self._owns_line and cells > small:
            self._visualizer.draw_tall(frame, 0, 0, cells)
            if self._visualizer_at_rest():
                self._line.move(dt, self._vis_at + self.width, 0.0)
            else:
                # Words going past over the top of it, the visualizer's own
                # top following them back into place.
                frame.paste(0, (0, 0, self.width, self.TITLE_ROWS))
                self._draw_title(frame, dt)
        elif self._owns_line:
            self._visualizer.settle()
            self._draw_title(frame, dt)
        if area is not None:
            rows = -(-cells // SoundMedium.SCALE)
            frame.paste(area, (0, rows + self.ART_TOP - self.TITLE_ROWS))
        return frame

    def _over(self, frame, shift, backdrop):
        """The frame short of home by `shift` rows, over what it covers.

        With a ticker in the backdrop only the cover rises, under title rows
        that are the ticker's or, once the line is this one's, the title's.
        Without one, the title rises glued to the cover. Rows the backdrop
        marks as pulled ride along just above the cover, as if it were pushing
        them up, as they do for the overview.

        The gauges are drawn onto a copy. They arrive as whatever the
        compositor happens to hold and pasting into that would leave the
        picture in it, to be redrawn as part of the gauges a frame later.
        """
        base, pulled, ticker = backdrop
        base = base.copy() if base.mode == 'L' else base.convert('L')
        bands = [(top, base.crop((0, top, self.width, top + rows)))
                 for top, rows in pulled]
        rise = self.TITLE_ROWS if ticker is not None else 0
        if self.height - shift > rise:
            base.paste(frame.crop((0, rise, self.width, self.height - shift)),
                       (0, rise + shift))
        if rise and self._owns_line:
            base.paste(frame.crop((0, 0, self.width, rise)), (0, 0))
        for top, band in bands:
            y = shift + self.ART_TOP - band.height
            if y <= top:
                base.paste(band, (0, y))
        return base

    def _clear(self):
        self._key, self._parting, self._ticker = None, None, None
        self._slide_pos, self._slide_at = 0.0, None
        self._owns_line = False
        self._side, self._pans, self._still = None, 0, 0.0
        self._phase, self._song, self._grow = 'art', None, 0.0
        self._last_area = None
        self._outro = False
        self._unending_at = None
        self._held = None
        self._visualizer.settle()

    def render(self, under=None):
        """A full-panel frame, or None to let the gauges through.

        `under` is an optional callable returning what render_backdrop in
        main.py does for the gauges alone - the composed frame, the rows to
        pull, and their ticker - given any words to hand that ticker first.
        A callable rather than an image because only the frames of a slide
        need it, and rendering the gauges beneath a settled cover would be
        work thrown away.
        """
        now = time.monotonic()
        dt = 0.0 if self._clock is None else min(max(now - self._clock, 0.0), 0.25)
        self._clock = now

        # No strip means the art is not wanted, or not ready past what
        # _current holds over - a remote cover that failed to download, say.
        # Either way there is nothing to draw, so it leaves the panel.
        state, strip = self._current()

        if strip is None:
            if self._slide_pos <= 0.0 or self._parting is None:
                if self._owns_line:
                    self._let_go(under, dt)
                self._clear()
                return None
            # The line goes back before the gauges draw, so their ticker
            # carries on from it this very frame.
            backdrop = self._let_go(under, dt)
            self._move_slide(-1)
            shift = self._shift()
            if shift >= self.height:
                # Clear of the bottom edge; the gauges have the panel back.
                self._clear()
                return None
            return self._over(self._parting, shift, backdrop)

        if state.path != self._key:
            # A new cover. Start from its edge, not from wherever the last
            # one had got to.
            self._key = state.path
            self._pan, self._direction = 0.0, 1
            self._panned_at = time.monotonic()
        self._follow_song(state)

        self._move_slide(+1)
        shift = self._shift()
        backdrop = None
        if shift > 0:
            backdrop = self._backdrop(under)
            if not self._owns_line and backdrop[2] is None:
                # Nothing to share the row with: the row rises with the cover.
                self._own_line(None)
        elif not self._owns_line:
            # Arrived under the ticker: its line becomes this one's, from
            # where it was drawn last frame, and moves on from there this one.
            self._own_line(self._ticker)

        self._move_grow(dt)
        area = self._cycle(state, strip, dt)
        self._last_area = area
        frame = self._compose(area, dt)
        # Kept whole to slide out with, still, should playback stop.
        self._parting = frame
        return frame if shift <= 0 else self._over(frame, shift, backdrop)
