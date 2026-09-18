import json
import math
import os
import time
from collections import namedtuple

import numpy as np
from PIL import Image

from utils.ticker_line import TickerLine
from utils.tiny_font import (TINY_HEIGHT, sanitize_tiny_text,
                             tiny_text_width)

# A window as the panel draws it: an inclusive pixel rectangle inside its
# workspace's thumbnail. `rank` is recency, 0 being the window last used there.
Window = namedtuple('Window', 'id x0 y0 x1 y1 rank title')
# One read of what the shell published. `position` is the view's place between
# workspaces, fractional mid-swipe; `active` is the workspace it will land on.
# `progress` is how far the overview has come up, 0 to 1, or None from a shell
# that does not say, in which case the panel slides in on a timer instead.
# `hovered` is the id of the window the pointer last moved onto, or None.
Snapshot = namedtuple('Snapshot',
                      'active position progress thumb_height workspaces hovered')


class OverviewOverlay:
    """The GNOME overview, drawn as window outlines.

    Not a module, for the same reason ArtworkOverlay is not: it takes all 34
    rows while the overview is open, so main.py asks it for a frame before
    anything else.

    The panel is the overview turned on its end. Every workspace is a thumbnail
    the panel's full 9 columns wide and as tall as the screen's shape makes
    that - 6 rows for a 16:10 screen - stacked down the panel a row apart, so
    four of them fill the 28 rows under the title. In each, every window the
    overview shows is a dim grey panel with a brighter outline, painted from
    the bottom of the stack up, so a window on top hides what it covers the way
    it hides the window on screen. Black is left for the desktop alone: a
    thumbnail's background, and the gap between thumbnails. A faint pixel
    marks each corner of the screen, which is the only thing an empty
    workspace draws, and gives a lone small window something to be positioned
    against.

    What is in view is framed: a line across the gap row above the workspace
    being looked at and another across the gap below, with everything between
    them at full brightness and the rest of the stack dimmed, the way the shell
    dims the workspaces beside the one in view. The frame is placed from the
    shell's workspace adjustment rather than the active index, which is
    fractional while a swipe is under way, so it travels down the stack a whole
    row at a time under the finger, cutting across both thumbnails mid-swipe
    exactly where the view is, and a swipe abandoned halfway goes back the same
    way. A frame rather than cross-fading the two thumbnails' brightness: a
    fade says roughly that a switch is happening, but not how far along it is,
    and a line's position can be read at a glance. Past four workspaces the
    stack scrolls to keep the frame in view.

    Its lines march like a marquee selection: bright heads, each with a tail
    fading to dark behind it, running right along the top line and left along
    the bottom, so the pattern goes clockwise round the view. Mid-swipe a line
    cuts across thumbnails, and a solid grey one read as just another window
    edge there. Nothing else in the overview moves sideways like that, and
    each head is brighter and each gap darker than whatever it crosses, so
    the line stays recognisable over outlines, fills and the desktop alike.
    Heads move in fractions of a pixel, fading in over the pixel ahead of
    them, so the march glides rather than stepping.

    The title row is the ticker's, in the ticker's place: the top five rows
    scroll the name of a window on the workspace being switched to, the way
    the transit incidents scroll there the rest of the time. First the
    workspace's number comes in and rests in the middle, and stays there.
    Titles are not cycled through: moving the pointer onto a window in the
    overview names that one, hurrying off what is on the line the way a
    switch does, and its title goes past twice before the number comes back
    to rest. Whichever window is being named - or while the number shows, the
    one last used there - is lifted to the top
    of its thumbnail so none of it is hidden, and pulses: the whole window
    brightens from how every other window looks, outline up to full and fill
    a little, and back again. That link is the point of the pulse: a 9px
    outline cannot say which window it is, and a title cannot say where, so
    each answers for the other. An empty workspace has no title to give, and
    shows its number.

    A pulse rather than anything that switches pixels off. In one bit this was
    a two-pixel gap chasing round a solid line, because half-lit dashes
    dissolved the shape they marked - tiled windows share their dividing
    column, so the divider flickered away. The pulse only varies how bright
    the window is, so the rectangle never breaks. It is the change that marks
    the window, not a brighter outline: kept above every other outline, the
    highlight had too little range left at the top to be seen. And the whole
    window at once, rather than crests travelling round its outline, which
    read as another marquee beside the frame's.

    The highlight follows the view, like the brightness does, rather than the
    workspace the title belongs to. The shell only changes its active
    workspace once a swipe has settled, and a highlight that waited for that
    left the window just swiped away from lit, then dimmed it and lit the new
    one a beat after the frame had arrived. Instead every workspace the view
    is within one of has a window highlighted - the one being named, or where
    nothing is yet, the one that will be first - by as much as the view is
    on that workspace, so one fades out as the frame leaves and the other in
    as it arrives, and nothing is left to change when the shell catches up.

    It rises from below over the gauges as the overview opens and drops away
    as it closes, following the shell's own progress, so a swipe that goes
    halfway and back down is followed the same way the brightness follows a
    swipe between workspaces. A shell that does not publish progress gets a
    timed slide instead, as the artwork does.

    Only the stack rises, with the workspace row riding just above it. The
    ticker stays where it is and keeps scrolling until the stack arrives under
    it, and then the line changes hands at a word boundary rather than being
    cut: the words already on the panel carry on past, and the title comes in
    where the next word would have, the words ahead of it hurried off so it
    is not kept waiting. Switching workspace hands one title on to the next
    the same way. Closing gives the line back to the ticker, and each resumes
    at the word it was interrupted before if the line comes back to it. Over the
    artwork the track's title is the ticker, and is shared the same way. With
    no ticker at all, the title rises glued to the stack instead.
    """

    DEFAULT_STATE_FILE = os.path.expanduser('~/.cache/pixeled/overview')
    # Seconds. Faster than the other state files, because the adjustment is
    # published at ~30Hz mid-swipe and the highlight should not lag the finger
    # by a poll. The file is only re-parsed when it has been replaced.
    POLL_INTERVAL = 0.03
    TITLE_ROWS = TINY_HEIGHT        # rows 0-4, where the transit ticker scrolls
    STACK_TOP = TITLE_ROWS + 1      # one blank row between title and stack
    MIN_THUMB_ROWS = 3
    # Seconds. The highlight eases toward the published position with this
    # time constant, which is what turns ~30Hz updates into ~60fps movement.
    # Short enough not to read as lag; the shell already eases it itself.
    FOLLOW_TAU = 0.04
    SLIDE_TIME = 0.2                # seconds to rise over the gauges, or fall off
    # Brightness, 0-255. Outlines well above fills so a window reads as its
    # edge first; fills well above black so overlapping windows stay distinct
    # from the desktop between them.
    #
    # Ordinary windows are kept well down the range so the named one has room
    # above them. The panel's brightness is linear in the value sent, and the
    # eye is not: an outline at 150 already looks nearly as bright as 255, and
    # a named window lit only as far above that as the range allowed could
    # not be told from its neighbours.
    WINDOW_FILL = 30
    WINDOW_OUTLINE = 90
    # The pulse's peak. It starts from WINDOW_FILL and WINDOW_OUTLINE, and the
    # fill's peak stays below the outline's start so the edge never dissolves.
    NAMED_FILL = 50
    NAMED_OUTLINE = 255
    CORNER = 64
    INACTIVE = 0.4                  # brightness of what is outside the frame
    # The marquee on the lines above and below what is in view.
    FRAME_PEAK = 255
    FRAME_PERIOD = 5.0              # px from one head to the next
    FRAME_TAIL = 3.5                # px a head's trail takes to fade to dark
    FRAME_SPEED = 8.0               # px/s
    PULSE_PERIOD = 1.0              # seconds from one peak to the next
    # px/s, overridden at runtime by pixeled-speed along with the ticker's.
    TEXT_SPEED = 24.0
    EPSILON = 1e-6                  # keeps an edge exactly on a column boundary
                                    # from spilling into the next column

    def __init__(self, width, height, state_file=None):
        self.width = width
        self.height = height
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self._line = TickerLine(width, self.TEXT_SPEED)
        self._next_poll = 0.0
        self._file_key = None           # (mtime, inode, size) last parsed
        self._snapshot = None
        self._shown = None              # last snapshot drawn, kept to slide out
        self._slide = 0.0               # 0 is clear of the bottom edge, 1 home
        self._position = None           # workspace in view, eased and fractional
        self._clock = None
        self._pulse = 0.0               # seconds the pulse has run
        self._marquee = 0.0             # px the frame's heads have travelled
        self._home = False              # whether the last frame was fully up
        self._ticker = None             # the backdrop's ticker, last slide frame
        self._reset_line()

    def set_scroll_speed(self, px_per_second):
        """Retune the title's scroll rate while running."""
        self._line.text_speed = float(px_per_second)

    # ── State ────────────────────────────────────────────────────────────────

    def _read_state(self, now):
        """Latest snapshot while the overview is open, otherwise None.

        A bad or missing read clears, as the artwork's does: this covers the
        gauges, so a shell that stops publishing has to give the panel back.
        """
        if now < self._next_poll:
            return self._snapshot
        self._next_poll = now + self.POLL_INTERVAL

        try:
            stat = os.stat(self.state_file)
        except OSError:
            self._file_key, self._snapshot = None, None
            return None
        # The extension replaces the file by rename, so any rewrite changes
        # this; an unchanged file does not need reading, let alone parsing.
        key = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
        if key == self._file_key:
            return self._snapshot
        self._file_key = key

        try:
            with open(self.state_file) as f:
                self._snapshot = self._parse(json.load(f))
        except (OSError, ValueError, KeyError, TypeError):
            self._snapshot = None
        return self._snapshot

    def _parse(self, data):
        if not data.get('visible'):
            return None
        raw = data.get('workspaces') or []
        if not raw:
            return None

        aspect = float(data.get('aspect') or 0.625)
        rows = round(self.width * aspect)
        thumb_height = max(self.MIN_THUMB_ROWS,
                           min(rows, self.height - self.STACK_TOP))

        count = len(raw)
        active = max(0, min(int(data.get('active', 0)), count - 1))
        position = float(data.get('position', active))
        position = max(0.0, min(position, count - 1.0))
        progress = data.get('progress')
        if progress is not None:
            progress = max(0.0, min(float(progress), 1.0))

        workspaces = [self._windows(windows, thumb_height) for windows in raw]
        hovered = data.get('hovered')
        return Snapshot(active, position, progress, thumb_height, workspaces,
                        hovered)

    def _windows(self, windows, rows):
        """A workspace's windows as pixel rectangles, bottom of the stack first.

        Edges round outward, so a window never shrinks to nothing and two
        windows tiled side by side share the column their edges meet in -
        which draws as one dividing line, the way the split looks on screen.
        """
        out = []
        for win in windows:
            x, y = float(win['x']), float(win['y'])
            right, bottom = x + float(win['w']), y + float(win['h'])
            if right <= 0 or bottom <= 0 or x >= 1 or y >= 1:
                continue    # entirely off the primary monitor
            x0, x1 = self._span(x, right, self.width)
            y0, y1 = self._span(y, bottom, rows)
            out.append(Window(win.get('id'), x0, y0, x1, y1,
                              int(win.get('rank', 0)),
                              sanitize_tiny_text(win.get('title') or '')))
        return out

    def _span(self, start, end, size):
        first = math.floor(start * size + self.EPSILON)
        last = math.ceil(end * size - self.EPSILON) - 1
        first = max(0, min(first, size - 1))
        last = max(first, min(last, size - 1))
        return first, last

    # ── Title ────────────────────────────────────────────────────────────────

    # The title row is a TickerLine: words carried over from what was showing,
    # then the workspace's number, which comes to rest in the middle and stays.
    # Hovering a window puts its title on after the number, once through, and
    # then the number comes back to rest.

    def _reset_line(self):
        self._line.reset()
        self._title_at = 0              # u the current title starts at
        self._title_skip = 0            # characters of it already said
        self._label_at = 0              # u of the workspace number
        self._numbering = False         # whether the number has the line
        self._placed = False            # whether anything has been put on it
        self._resume = None             # (workspace, window, skip) to go back to
        self._title_workspace = None    # workspace the title belongs to
        self._title_window = None       # id of the window being named, or None
        self._hovered = None            # the hover last seen, acted on or not

    def _take_line(self, ticker, snapshot):
        """Carry on from the ticker's words, the title in place of its next.

        If this is the stack coming back up after starting to leave, a hovered
        window's title picks up at the word it was interrupted before, as the
        ticker does.
        """
        start = self._line.take(ticker)
        self._placed = True
        resume, self._resume = self._resume, None
        ids = [w.id for w in snapshot.workspaces[snapshot.active]]
        if resume is not None and resume[0] == snapshot.active and resume[1] in ids:
            self._title_workspace, self._title_window, self._title_skip = resume
            self._title_at = start
            self._numbering = False
        else:
            self._title_workspace = None    # _named starts afresh after them

    def _give_line(self, snapshot, dt):
        """The words on the panel, for the ticker to carry on from.

        Also notes where a title was cut, so a swipe back up resumes it there.
        If all of it had already come on, or the number had the line, the
        number comes back instead.
        """
        handback, cut = self._line.give(dt)
        drawn = self._line.drawn
        self._resume = None
        if (self._title_window is not None and drawn and not self._numbering
                and cut is not None):
            skip = self._title_skip
            if cut[0] == len(drawn) - 1:
                skip += cut[1]
            self._resume = (snapshot.active, self._title_window, skip)
        return handback

    def _say_number(self, snapshot):
        """Put the workspace's number on after the words on the panel, to come
        to rest in the middle."""
        line = self._line
        start = line.follow_on()
        if self._placed:
            self._label_at = start
        else:
            # First thing on a line just opened, with the title row rising
            # along with the stack: already in place.
            self._label_at = (math.floor(line.scroll) - self.width
                              + self._label_x(snapshot))
        self._numbering, self._placed = True, True
        self._title_window, self._title_skip = None, 0

    def _named(self, snapshot):
        """The window to highlight, and whose title is scrolling if one is.

        Follows `active`, not the view: mid-swipe that is between two
        workspaces and the title would restart every time the finger wobbled
        across the middle. `active` changes once, when a swipe settles. The
        highlight follows the view instead; see _draw_stack.

        While the number is showing this is the window last used there.
        """
        line = self._line
        windows = sorted(snapshot.workspaces[snapshot.active],
                         key=lambda w: w.rank)
        ids = [w.id for w in windows]
        if (snapshot.active != self._title_workspace
                or (self._title_window is not None
                    and self._title_window not in ids)):
            # A new workspace, or the window being named has closed: say
            # which workspace this is, after the words on the panel.
            self._title_workspace = snapshot.active
            self._say_number(snapshot)
        if not windows:
            return None

        # Moving the pointer onto a window names it, after the words on the
        # panel, hurrying them off the way arriving at a workspace does. Acted
        # on only when the hover changes, so a pointer resting on a window
        # neither restarts its title nor replaces the number of a workspace
        # switched to under it.
        hovered = snapshot.hovered
        if hovered != self._hovered:
            self._hovered = hovered
            if hovered in ids and hovered != self._title_window:
                self._title_window, self._title_skip = hovered, 0
                self._title_at = line.follow_on()
                self._numbering = False

        if self._numbering:
            return windows[0]
        window = windows[ids.index(self._title_window)]
        text = window.title[self._title_skip:]
        if line.x(self._title_at) + tiny_text_width(text) <= 0:
            self._say_number(snapshot)
            return windows[0]
        return window

    def _label_x(self, snapshot):
        """Where the workspace number comes to rest, centred."""
        return (self.width - tiny_text_width(str(snapshot.active + 1))) // 2

    def _move_line(self, snapshot, dt):
        """Scroll the line toward the number reaching the middle at rest, or
        the title reaching the right edge at reading speed."""
        line = self._line
        if self._numbering:
            line.move(dt, self._label_at + self.width - self._label_x(snapshot), 0.0)
        else:
            line.move(dt, self._title_at, line.text_speed)

    def _draw_title(self, image, snapshot, window):
        if self._numbering:
            placed = [(str(snapshot.active + 1), self._label_at)]
        else:
            placed = [(window.title[self._title_skip:], self._title_at)]
        self._line.draw(image, placed)

    # ── Stack ────────────────────────────────────────────────────────────────

    def _outline(self, canvas, win, level):
        canvas[win.y0, win.x0:win.x1 + 1] = level
        canvas[win.y1, win.x0:win.x1 + 1] = level
        canvas[win.y0:win.y1 + 1, win.x0] = level
        canvas[win.y0:win.y1 + 1, win.x1] = level

    def _thumbnail(self, windows, rows, named, weight):
        """A workspace's windows, `named` highlighted by `weight`, 0 to 1."""
        canvas = np.zeros((rows, self.width))
        if weight <= 0:
            named = None
        if named is not None and weight > 0.5:
            # Lifted to the top, so the window being named is never hidden.
            # Only while the view is mostly here: restacking is a jump, so it
            # happens mid-swipe, with everything moving, and not as the view
            # eases the last fraction of the way away and the shell settles.
            windows = [w for w in windows if w.id != named.id] + [named]
        # How far through the pulse the window is, 0 as any other window to 1
        # at the peak, scaled down while the view is only partly here.
        lift = weight * (0.5 + 0.5 * math.cos(2 * math.pi * self._pulse
                                              / self.PULSE_PERIOD))
        for win in windows:
            if named is not None and win.id == named.id:
                canvas[win.y0:win.y1 + 1, win.x0:win.x1 + 1] = (
                    self.WINDOW_FILL + (self.NAMED_FILL - self.WINDOW_FILL) * lift)
                self._outline(canvas, win, self.WINDOW_OUTLINE
                              + (self.NAMED_OUTLINE - self.WINDOW_OUTLINE) * lift)
            else:
                canvas[win.y0:win.y1 + 1, win.x0:win.x1 + 1] = self.WINDOW_FILL
                self._outline(canvas, win, self.WINDOW_OUTLINE)
        for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            # Under a window reaching the corner, its own outline is the mark.
            canvas[y, x] = max(canvas[y, x], self.CORNER)
        return canvas

    def _draw_stack(self, canvas, snapshot, named):
        rows = snapshot.thumb_height
        pitch = rows + 1
        area = self.height - self.STACK_TOP
        total = len(snapshot.workspaces) * pitch - 1

        # Stack row the view starts at, in whole rows, so the frame steps
        # rather than smearing across two rows mid-swipe.
        view = round(self._position * pitch)
        # Keep the view centred once the stack is taller than the panel, and
        # the stack pinned to the top while it is not.
        camera = max(0, min(view - (area - rows) // 2, max(0, total - area)))

        stack = np.zeros((max(total, area) + pitch, self.width))
        for i, windows in enumerate(snapshot.workspaces):
            top = i * pitch
            if top + rows <= camera or top >= camera + area:
                continue
            # By how much the view is on this workspace, and the window that
            # is, or would be, named there: the title only moves to another
            # workspace once the shell activates it, and then starts with the
            # window last used there.
            weight = max(0.0, 1.0 - abs(self._position - i))
            if i == snapshot.active:
                here = named
            else:
                here = min(windows, key=lambda w: w.rank, default=None)
            stack[top:top + rows] = self._thumbnail(windows, rows, here, weight)
        dim = np.full((stack.shape[0], 1), self.INACTIVE)
        dim[max(view, 0):view + rows] = 1.0
        stack *= dim

        canvas[self.STACK_TOP:] = stack[camera:camera + area]
        # The frame's lines sit in the gap rows either side of the view. The
        # top workspace's upper line is the blank row under the title, so the
        # lines are placed on the canvas rather than in the stack.
        for row, line in ((view - 1, self._marquee_line(1)),
                          (view + rows, self._marquee_line(-1))):
            y = self.STACK_TOP + row - camera
            if self.TITLE_ROWS <= y < self.height:
                canvas[y] = line

    def _marquee_line(self, direction):
        """One frame line's brightness, its heads travelling in `direction`.

        Measured from the head each pixel trails, a pixel is lit fully at the
        head, fades to dark over the tail, and is dark until the next head is
        within a pixel, over which it fades back in. The bottom line is the
        top one mirrored, so a head leaving one line's end enters the other's.
        """
        x = np.arange(self.width, dtype=float)
        if direction < 0:
            x = x[::-1]
        # Distance behind the nearest head, from -1 (a pixel ahead) upwards.
        behind = (self._marquee - x + 1.0) % self.FRAME_PERIOD - 1.0
        level = np.where(behind < 0.0, 1.0 + behind,
                         np.clip(1.0 - behind / self.FRAME_TAIL, 0.0, 1.0))
        return self.FRAME_PEAK * level

    # ── Frame ────────────────────────────────────────────────────────────────

    def _frame(self, snapshot, dt):
        target = snapshot.position
        if self._position is None:
            self._position = target
        else:
            self._position += (target - self._position) * (
                1 - math.exp(-dt / self.FOLLOW_TAU))

        named = self._named(snapshot)
        self._move_line(snapshot, dt)
        self._pulse = (self._pulse + min(dt, 0.25)) % self.PULSE_PERIOD
        self._marquee += self.FRAME_SPEED * min(dt, 0.25)

        canvas = np.zeros((self.height, self.width))
        self._draw_stack(canvas, snapshot, named)
        image = Image.fromarray(np.rint(canvas).astype(np.uint8), 'L')
        self._draw_title(image, snapshot, named)
        return image

    def _over(self, frame, shift, under, handback=None):
        """The frame short of home by `shift` rows, over what it covers.

        With a ticker in the backdrop, the title rows are left to it and only
        the stack rises; without one, the title rises glued to the stack.
        Rows the backdrop marks as pulled ride along just above the stack, as
        if it were pushing them up. They wait at home until the stack reaches
        them, and are gone once it is home, where they would close the gap
        under the title.
        """
        base, pulled, ticker = (under(handback) if under is not None
                                else (None, (), None))
        if base is None or base.size != (self.width, self.height):
            base, pulled, ticker = Image.new('L', (self.width, self.height), 0), (), None
        self._ticker = ticker
        base = base.copy() if base.mode == 'L' else base.convert('L')
        bands = [(top, base.crop((0, top, self.width, top + rows)))
                 for top, rows in pulled]
        rise = self.TITLE_ROWS if ticker is not None else 0
        if self.height - shift > rise:
            base.paste(frame.crop((0, rise, self.width, self.height - shift)),
                       (0, rise + shift))
        for top, band in bands:
            y = shift + self.STACK_TOP - band.height
            if y <= top:
                base.paste(band, (0, y))
        return base

    def render(self, under=None):
        """A full-panel frame, or None while the overview is closed.

        `under` is a callable returning what the overview slides over, the
        (top, height) of any rows in it to pull up with the stack, and the
        ticker in it the title shares its row with, or None. It is called only
        on the frames of the slide itself, with the title's words to give back
        to that ticker on the first frame out, before anything is drawn.
        """
        now = time.monotonic()
        dt = 0.0 if self._clock is None else now - self._clock
        self._clock = now

        snapshot = self._read_state(now)
        if snapshot is not None:
            if self._shown is None:
                # Opening from nothing: the highlight starts where the view is,
                # rather than easing over from where the last visit left it.
                self._position = None
                self._reset_line()
            self._shown = snapshot
            if snapshot.progress is None:
                self._slide = min(1.0, self._slide + min(dt, 0.05) / self.SLIDE_TIME)
            else:
                # Eased like the highlight, which turns the ~30Hz updates into
                # ~60fps movement.
                self._slide += (snapshot.progress - self._slide) * (
                    1 - math.exp(-min(dt, 0.05) / self.FOLLOW_TAU))
        else:
            self._slide = max(0.0, self._slide - min(dt, 0.05) / self.SLIDE_TIME)
            if self._shown is None:
                self._slide, self._clock = 0.0, None
                return None

        shift = round(self.height * (1 - self._slide))
        if shift >= self.height:
            # Nothing of the overview is on the panel - closed, or open but
            # held below the bottom edge mid-swipe - so leave the panel to
            # whatever is underneath.
            if snapshot is None:
                self._slide, self._shown, self._clock = 0.0, None, None
                self._ticker = None
            self._home = False
            return None

        if shift <= 0 and not self._home and self._ticker is not None:
            # Arrived under the ticker: its line becomes the title's, from
            # where it was drawn last frame, and moves on from there this one.
            self._take_line(self._ticker, self._shown)
        frame = self._frame(self._shown, dt)
        if shift <= 0:
            self._home = True
            return frame
        # Leaving: the line goes back to the ticker from where this frame has
        # it, before the backdrop draws, so it carries on without a stall.
        handback = self._give_line(self._shown, dt) if self._home else None
        self._home = False
        return self._over(frame, shift, under, handback)
