import json
import math
import os
import time
from PIL import Image
from modules.module_base import ModuleBase


class WorkspaceModule(ModuleBase):
    """A one-pixel-tall GNOME workspace indicator.

    Drawn like the shell's own workspace list wherever it fits: dots, one per
    workspace, separated by a blank column, with the active workspace widened
    into a pill the way GNOME widens the active thumbnail, and drawn at full
    brightness against dimmed dots, which is also how the shell draws its
    inactive workspaces. Shape and brightness both carry the marker, so nothing
    blinks - the row is a still picture that only changes when the workspace
    does.

    The pill is 3px, not 2px. A 2px pill reads as a dot with a pixel stuck on
    one side, and which side depends on where it sits: only the leftmost
    workspace has a spare column to grow into, so every other pill has to take
    its extra column from a gap and slide the dots past it along, which looks
    like the pixel was added on the other side. 3px is a run rather than a
    fattened dot, so it reads as one mark at any position. It costs nothing: a
    2px pill still needs its separator, so both spend two columns per
    workspace and the group spans 2*count+1 either way - exactly nine at four
    workspaces, which is the most that fit regardless.

    The group is left justified, as the bar below is, so the dots keep a pitch
    of two columns from a fixed left edge whatever the count.

    That layout costs two columns per workspace (dot plus gap, and the pill
    spends its gap on a second lit pixel), so nine columns hold four
    workspaces. At five the pills would have to touch and the row would read as
    one blob, so past that the module switches to a bar: `count` dim pixels
    with no gaps, left justified, capped at the nine the panel has, and the
    active one at full brightness. The bar gives up on being countable at a
    glance - what survives is roughly how many and, from the marker, which one.

    Between workspaces the row follows the view the way the overview's frame
    does: the shell publishes where the view is, fractional while its switch
    animation runs, whether that was a swipe or a keyboard shortcut, and the
    row eases toward it every frame. The pill glides in fractions of a column,
    its light shared between the columns either side of where it really is.
    The dots never move or fade: the pill slides over them, uncovering the dot
    of the workspace being left and covering the one it arrives at, so every
    workspace keeps its place and no column dims on the way. The bar's marker
    glides the same way, from one cell's brightness to the next.

    That wants the patched firmware's ~60fps; on stock firmware's ~6fps it
    catches a few intermediate frames of a switch rather than a glide.
    """

    DEFAULT_STATE_FILE = os.path.expanduser('~/.cache/pixeled/workspaces')
    # Seconds. As fast as the overview's, because the position is published at
    # ~30Hz while a switch animates. The file is only re-parsed when replaced.
    POLL_INTERVAL = 0.03
    # Seconds. The row eases toward the published position with this time
    # constant, as the overview's frame does, turning ~30Hz updates into
    # ~60fps movement. A shell that publishes no position gets the same glide
    # from one active workspace to the next.
    FOLLOW_TAU = 0.04
    INACTIVE_GREY = 96    # dim level for inactive workspaces
    PILL = 3              # columns the active workspace's pill spans

    def __init__(self, height=1, state_file=None, hide_trailing_empty=True):
        # height may arrive as None: main.py passes config.json's "height"
        # through unconditionally, absent or not.
        super().__init__(height or 1)
        self.state_file = state_file or self.DEFAULT_STATE_FILE
        self.hide_trailing_empty = hide_trailing_empty
        self._state = None      # last good (position, count), or None
        self._next_poll = 0.0
        self._file_key = None   # (mtime, inode, size) last parsed
        self._shown = None      # position drawn, eased toward the published one
        self._clock = None

    def _read_state(self):
        """Latest (position, count) from the shell extension, or None.

        Polled on a cadence rather than every frame, and a bad read keeps the
        previous value: the file is replaced by rename, but it is also absent
        until the extension first runs, and the panel should not flicker to
        blank whenever the shell restarts.
        """
        now = time.monotonic()
        if now < self._next_poll:
            return self._state
        self._next_poll = now + self.POLL_INTERVAL

        try:
            stat = os.stat(self.state_file)
            key = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
            if key == self._file_key:
                return self._state
            with open(self.state_file) as f:
                data = json.load(f)
            count = int(data['count'])
            active = int(data['active'])
            position = float(data.get('position', active))
        except (OSError, ValueError, KeyError, TypeError):
            return self._state
        self._file_key = key

        if count < 1 or position != position:
            return self._state
        active = max(0, min(active, count - 1))
        position = max(0.0, min(position, count - 1.0))

        # Drop the spare workspace dynamic mode keeps at the end, so the row
        # counts what is actually in use and stops growing and shrinking as
        # windows move. Not while standing on it, or on the way to it, though -
        # switching to the spare is how you create a workspace, and the
        # indicator has to show where you are.
        if (self.hide_trailing_empty and data.get('trailing_empty')
                and count > 1 and active != count - 1
                and position <= count - 2):
            count -= 1

        self._state = (position, count)
        return self._state

    def _follow(self, position, count):
        """The position to draw this frame, eased toward the published one."""
        now = time.monotonic()
        dt = 0.0 if self._clock is None else min(max(now - self._clock, 0.0), 0.25)
        self._clock = now
        if self._shown is None:
            self._shown = position
        else:
            self._shown += (position - self._shown) * (
                1 - math.exp(-dt / self.FOLLOW_TAU))
        # A count that shrank under the pill leaves nothing to glide from.
        self._shown = max(0.0, min(self._shown, count - 1.0))
        return self._shown

    @staticmethod
    def _fits(width, count):
        """Whether count workspaces fit as dots with the active one widened.

        Each workspace takes a column plus a separating one, and the active
        workspace spends its separator plus one more column on the pill, so
        the group spans 2*count+1 columns however the active one falls.
        """
        return 2 * count + 1 <= width

    def _dots(self, width, position, count):
        """The shell's layout: spaced dots, the view's place a pill.

        Settled on workspace a, the dots before it sit at 2i, the pill spans
        2a to 2a+2, and the dots after it at 2i+2. Mid-move toward a+1 the
        same holds, with workspace a's own dot at 2a: the pill still covers it
        at the start, and slides on to cover a+1's at 2a+4 by the end, which is
        exactly the settled layout on a+1. Drawn under the pill, then, the dots
        stay put and only the pill moves.
        """
        row = [0.0] * width
        a = min(int(position), count - 1)
        for i in range(count):
            x = 2 * i if i <= a else 2 * i + 2
            if 0 <= x < width:
                row[x] = float(self.INACTIVE_GREY)

        start = 2 * position
        for x in range(width):
            covered = min(x + 1, start + self.PILL) - max(x, start)
            if covered > 0:
                row[x] = max(row[x], 255 * min(covered, 1.0))
        return row

    def _bar(self, width, position, count):
        """Too many to space out: a solid left-justified run, active marked."""
        # More workspaces than columns: show a window around the active one
        # rather than dropping workspaces off the end. The bar looks the same
        # either way at this point - it fills the row - but the marker keeps
        # tracking where in the set you are.
        if count > width:
            start = min(max(round(position) - width // 2, 0), count - width)
            position -= start
            count = width

        row = [float(self.INACTIVE_GREY)] * count + [0.0] * (width - count)
        a = min(int(position), count - 1)
        t = position - a
        row[a] = self.INACTIVE_GREY + (255 - self.INACTIVE_GREY) * (1.0 - t)
        if a + 1 < count:
            row[a + 1] = self.INACTIVE_GREY + (255 - self.INACTIVE_GREY) * t
        return row

    def _row(self, width):
        """One row of brightness values."""
        state = self._read_state()
        if state is None:
            return [0] * width     # nothing published yet; draw nothing
        position, count = state
        position = self._follow(position, count)

        if self._fits(width, count):
            row = self._dots(width, position, count)
        else:
            row = self._bar(width, position, count)
        return [round(level) for level in row]

    def render(self, width):
        image = Image.new('L', (width, self.height), 0)
        row = self._row(width)
        for y in range(self.height):
            for x in range(width):
                image.putpixel((x, y), row[x])
        return image
