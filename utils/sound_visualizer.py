import copy
import ctypes
import math
import os
import re
import signal
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from utils.sound_medium import SoundMedium
from utils.volume_wedge import volume_wedge


class SoundVisualizer:
    """What the speakers are playing, as light moving in a little world.

    Something to put on the title row when there is nothing to say. It goes
    on a TickerLine like a word does - it scrolls in, rests, and scrolls out
    ahead of whatever is said next - so it has a width and draws itself at an
    x, which is all the line asks of a run that is not text.

    This half listens: it captures what is playing and hears it as bands,
    each with how loud it is, how suddenly it just got louder, and where it
    sits in the stereo image. SoundMedium is the other half, the world those
    push on and the light it gives off.

    Silence draws nothing of the world. Instead it shows the volume, as a
    wedge filled as far as the volume goes and slashed through when muted
    (see volume_wedge): dimly once nothing has been heard for IDLE_AFTER, and
    brightly for SHOW_FOR after the volume or mute changes, over the world
    dimmed under it if something is playing, faintly showing through the part
    of the wedge past the volume.

    It can also be drawn tall, for a view that gives it more of the panel: the
    same world, grown taller, with everything in it carried over, and shrunk
    back into the row afterwards (see draw_tall and settle).

    The audio is the playing sink's monitor, read with parec, whenever a sink
    is RUNNING - not only while this is on the panel. Started when it is drawn,
    the capture took a second or so to deliver, so the visualizer scrolled in
    blank and its bars popped up once it had stopped; running ahead of time,
    it moves in and out already showing the sound, as a word would. It does
    stop once this has gone unused for IDLE_STOP, for layouts that never
    show it.

    Only while a sink is RUNNING, because recording a monitor wakes the sink:
    capturing all the time would keep the sound card powered on a laptop that
    would otherwise suspend it seconds after the last sound. A sink that is
    already running costs nothing more to listen to. Our own capture leaves a
    sink IDLE, never RUNNING, so asking PipeWire whether one is running is not
    fooled by the capture itself.

    Playback starting or stopping is picked up from `pactl subscribe`, whose
    stream events come within a frame or two, and checked again shortly after
    each, since the sink's state can lag its streams. A slow poll backs that
    up in case an event is missed. The same events say when the default
    sink's volume may have changed, or which sink is the default; each is
    checked against the volume last read, since a sink also changes when it
    starts or stops playing.
    """

    RATE = 24000                # Hz; bands stop short of 12 kHz anyway
    FFT_SIZE = 1024             # samples; ~43 ms, 23 Hz a bin
    CHUNK = 256                 # frames a read; ~11 ms, so every frame is fresh
    BANDS = 12
    LOW_HZ, HIGH_HZ = 45.0, 11000.0
    POLL_INTERVAL = 10.0        # seconds between checks with no events
    SETTLE = (0.2, 1.0)         # seconds after an event to check again at
    IDLE_STOP = 300.0           # seconds undrawn before capture stops
    STALE = 0.25                # seconds without samples before it hears silence
    GATE_DB = -70.0             # dBFS; quieter than this is silence
    RANGE_DB = 36.0             # dB from an empty bar to a full one
    MIN_CEILING_DB = -36.0      # dBFS; keeps quiet passages from filling bars
    CEILING_DECAY = 4.0         # dB/s the loudness reference falls back
    TILT_DB = 2.5               # dB/octave, for music's falling treble
    # A band counts as struck by however much faster than this it got louder,
    # in its 0-1 loudness per second, so a swell pushes and an attack strikes.
    ONSET_RATE = 5.0
    RESET_GAP = 0.5             # seconds undrawn after which the world starts over
    # The volume.
    SHOW_FOR = 2.0              # seconds it stays up after the last change
    IDLE_AFTER = 10.0           # seconds of silence before it shows on its own
    IDLE_SHOW = 0.6             # how bright it shows then, of full
    SHOW_RISE, SHOW_FALL = 0.06, 0.4    # seconds it takes to come and go
    UNDER_DIM = 0.7             # how much of the world under it is dimmed

    def __init__(self, width, height=5):
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._buffer = np.zeros((self.FFT_SIZE, 2), dtype=np.float32)
        self._fresh_at = 0.0        # when samples last arrived
        self._drawn_at = time.monotonic()   # when last drawn; new counts as used
        self._wake = threading.Event()      # something to check the sinks for
        self._small = _View(SoundMedium(width, height, self.BANDS))
        self._tall = None           # the world grown taller, while it is
        self._loud = np.zeros(self.BANDS)
        self._ceiling = self.MIN_CEILING_DB
        self._heard_at = None
        self._heard = None          # (time, loud, onset, balance), for reuse
        self._sounding_at = -math.inf   # when anything was last heard
        self.volume = None          # (level, muted) of the default sink, once read
        self._volume_at = None      # when that last changed, or None
        self._volume_wake = threading.Event()
        self._shown = 0.0           # how brightly the volume shows, 0 to 1
        self._shown_clock = None
        self._shown_frame = None    # (time, shown, volume), for reuse
        self._thump = None          # a test strike for the next hearing

        edges = np.geomspace(self.LOW_HZ, self.HIGH_HZ, self.BANDS + 1)
        bins = np.fft.rfftfreq(self.FFT_SIZE, 1.0 / self.RATE)
        self._bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = np.nonzero((bins >= lo) & (bins < hi))[0]
            if idx.size == 0:
                # Narrower than a bin at the bottom end: take the nearest one.
                idx = np.array([np.argmin(np.abs(bins - (lo + hi) / 2))])
            self._bands.append(idx)
        centres = np.sqrt(edges[:-1] * edges[1:])
        self._tilt = self.TILT_DB * np.log2(centres / centres[0])
        self._window = np.hanning(self.FFT_SIZE).astype(np.float32)[:, None]
        # A full-scale sine's energy through the window, so levels read in dBFS.
        self._full_scale = (self._window.sum() / 2) ** 2 * 1.5
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._watch_events, daemon=True).start()
        threading.Thread(target=self._watch_volume, daemon=True).start()

    # ── Capture ──────────────────────────────────────────────────────────────

    @staticmethod
    def _env():
        # The service is a system unit, which has no session environment; the
        # user's sound server is still at the usual place under /run/user.
        env = dict(os.environ)
        env.setdefault('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
        return env

    @staticmethod
    def _die_with_parent():
        # So a parec left behind by a crashed or interrupted pixeled cannot
        # keep the sound card awake. PR_SET_PDEATHSIG is 1.
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)

    def _playing_sink(self):
        """The name of a sink something is playing to, or None."""
        try:
            out = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                                 capture_output=True, text=True, timeout=2,
                                 env=self._env()).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            fields = line.split('\t')
            if len(fields) >= 5 and fields[4].strip() == 'RUNNING':
                return fields[1]
        return None

    def _watch_events(self):
        """Wake the capture loop whenever a stream or sink changes."""
        while True:
            try:
                proc = subprocess.Popen(['pactl', 'subscribe'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True,
                                        env=self._env(),
                                        preexec_fn=self._die_with_parent)
            except OSError:
                return      # no pactl: the slow poll is all there is
            for line in proc.stdout:
                # "Event 'new' on sink-input #93": sink and sink-input events,
                # not the client events every pactl run itself makes.
                if " on sink" in line:
                    self._wake.set()
                # The sink's own, not its streams': volume and mute are on
                # the sink, and which sink is the default is on the server.
                if " on sink #" in line or " on server" in line:
                    self._volume_wake.set()
            proc.wait()
            self._wake.set()
            self._volume_wake.set()
            time.sleep(2.0)     # sound server gone; wait for it to come back

    def _capture_loop(self):
        proc, sink = None, None
        pending = []
        while True:
            wanted = time.monotonic() - self._drawn_at < self.IDLE_STOP
            playing = self._playing_sink() if wanted else None
            if proc is not None and (playing != sink or proc.poll() is not None):
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                proc = None
            if proc is None and playing is not None:
                sink = playing
                try:
                    proc = subprocess.Popen(
                        ['parec', '-d', f'{sink}.monitor', '--raw',
                         '--format=s16le', f'--rate={self.RATE}', '--channels=2',
                         '--latency-msec=20'],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        env=self._env(), preexec_fn=self._die_with_parent)
                except OSError:
                    proc = None
                else:
                    threading.Thread(target=self._read, args=(proc,),
                                     daemon=True).start()
            timeout = pending.pop(0) if pending else self.POLL_INTERVAL
            if self._wake.wait(timeout):
                self._wake.clear()
                pending = list(self.SETTLE)

    def _read_volume(self):
        """The default sink's (level, muted), 1 being 100%, or None."""
        try:
            volume, mute = (
                subprocess.run(['pactl', what, '@DEFAULT_SINK@'],
                               capture_output=True, text=True, timeout=2,
                               env=self._env()).stdout
                for what in ('get-sink-volume', 'get-sink-mute'))
        except (OSError, subprocess.SubprocessError):
            return None
        # "Volume: front-left: 27524 /  42% / -22.61 dB,   front-right: ..."
        channels = [int(raw) for raw in re.findall(r'(\d+) /\s*\d+%', volume)]
        if not channels or 'Mute:' not in mute:
            return None
        level = sum(channels) / len(channels) / 65536    # PA_VOLUME_NORM
        return round(level, 4), mute.split('Mute:')[1].strip() == 'yes'

    def _watch_volume(self):
        """Keep the volume current, noting when it changes. The first read,
        and the first after the sound server comes back, is not a change."""
        while True:
            volume = self._read_volume()
            if volume is not None and volume != self.volume:
                if self.volume is not None:
                    self._volume_at = time.monotonic()
                self.volume = volume
            self._volume_wake.wait(self.POLL_INTERVAL)
            self._volume_wake.clear()

    def _read(self, proc):
        size = self.CHUNK * 4
        while True:
            data = proc.stdout.read(size)
            if not data:
                self._wake.set()    # parec ended; see whether to restart it
                return
            chunk = np.frombuffer(data[:len(data) // 4 * 4], dtype='<i2')
            chunk = chunk.reshape(-1, 2).astype(np.float32) / 32768.0
            with self._lock:
                self._buffer = np.concatenate((self._buffer[len(chunk):], chunk))
                self._fresh_at = time.monotonic()

    # ── Picture ──────────────────────────────────────────────────────────────

    def _hear(self, samples, dt):
        """The sound as bands: how loud each is, 0 to 1, how sharply it just
        got louder, and where it sits between left (-1) and right (1)."""
        power = np.abs(np.fft.rfft(samples * self._window, axis=0)) ** 2
        left = np.array([power[idx, 0].sum() for idx in self._bands])
        right = np.array([power[idx, 1].sum() for idx in self._bands])
        energy = (left + right) / 2
        db = 10 * np.log10(np.maximum(energy / self._full_scale, 1e-12))
        db = np.where(db < self.GATE_DB, -np.inf, db + self._tilt)
        # The loudest band sets the top of the scale, following a louder
        # moment straight away and a quieter one slowly, so the world has the
        # same range to work with at any volume without pumping on every beat.
        self._ceiling = max(db.max(), self.MIN_CEILING_DB,
                            self._ceiling - self.CEILING_DECAY * dt)
        floor = self._ceiling - self.RANGE_DB
        loud = np.clip((db - floor) / self.RANGE_DB, 0.0, 1.0)
        onset = np.maximum(loud - self._loud - self.ONSET_RATE * dt, 0.0)
        self._loud = loud
        balance = (right - left) / np.maximum(right + left, 1e-12)
        return loud, onset, balance

    def _listen(self, now):
        """What the bands are doing this instant, worked out once for it."""
        if self._heard is not None and now - self._heard[0] < 0.005:
            return self._heard[1:]
        dt = 0.0 if self._heard_at is None else max(now - self._heard_at, 0.0)
        self._heard_at = now
        if dt > self.RESET_GAP:
            self._loud[:] = 0.0     # nothing to measure an onset against
            dt = 0.0
        with self._lock:
            samples, fresh_at = self._buffer, self._fresh_at
        if now - fresh_at < self.STALE:
            loud, onset, balance = self._hear(samples, min(dt, 0.25))
            if loud.any():
                self._sounding_at = now
        else:
            loud = onset = balance = np.zeros(self.BANDS)
            self._loud[:] = 0.0
        if self._thump is not None:
            onset, self._thump = np.maximum(onset, self._thump), None
        self._heard = (now, loud, onset, balance)
        return loud, onset, balance

    def thump(self, strength=0.6):
        """Strike the low bands as a hard kick drum would, on the next frame,
        to judge the world's settings by without waiting for one."""
        onset = np.zeros(self.BANDS)
        onset[:3] = strength
        self._thump = onset

    def _pixels(self, view):
        """A view's world run on to now, as rows x width of brightness 0-255.

        Worked out once per instant, since the panel can draw this twice in a
        frame - over and under an overlay mid-slide.
        """
        now = time.monotonic()
        medium = view.medium
        if (view.frame is not None and now - view.frame[0] < 0.005
                and view.frame[1] == medium.h):
            return view.frame[2]
        gap = 0.0 if view.clock is None else max(now - view.clock, 0.0)
        view.clock = now
        if gap > self.RESET_GAP:
            # Everything in it would have faded by now anyway.
            medium.reset()
            gap = 0.0
        medium.advance(gap, *self._listen(now))
        pixels = medium.light()
        view.frame = (now, medium.h, pixels)
        return pixels

    def _show_volume(self, now):
        """How brightly the volume shows this instant, and what it is.

        Coming up quickly and going slowly, toward full for SHOW_FOR after a
        change, and IDLE_SHOW once nothing has been heard for IDLE_AFTER.
        """
        if self._shown_frame is not None and now - self._shown_frame[0] < 0.005:
            return self._shown_frame[1:]
        volume = self.volume
        if volume is None:
            self._shown_frame = (now, 0.0, None)
            return 0.0, None
        changed = self._volume_at is not None and now - self._volume_at < self.SHOW_FOR
        idle = now - self._sounding_at >= self.IDLE_AFTER
        target = max(1.0 if changed else 0.0, self.IDLE_SHOW if idle else 0.0)
        gap = math.inf if self._shown_clock is None else max(now - self._shown_clock, 0.0)
        self._shown_clock = now
        if gap > self.RESET_GAP:
            self._shown = target    # it would have got there undrawn
        else:
            tau = self.SHOW_RISE if target > self._shown else self.SHOW_FALL
            self._shown += (target - self._shown) * (1 - math.exp(-gap / tau))
            if target == 0.0 and self._shown < 0.01:
                self._shown = 0.0
        self._shown_frame = (now, self._shown, volume)
        return self._shown, volume

    def _over(self, pixels):
        """The world's pixels with the volume over them, as it shows now.

        The wedge is the row's height however tall the world is drawn, so
        grown over the panel it stays at the top, where the row was, and only
        the world in those rows is dimmed under it.
        """
        shown, volume = self._show_volume(time.monotonic())
        if shown <= 0.0:
            return pixels
        rows = min(self.height, pixels.shape[0])
        light, cover = volume_wedge(rows, pixels.shape[1], *volume)
        top = pixels[:rows] * (1 - self.UNDER_DIM * shown) * (1 - cover * shown)
        top += 255 * (light * shown) ** SoundMedium.GAMMA
        out = pixels.copy()
        out[:rows] = np.rint(np.clip(top, 0, 255)).astype(np.uint8)
        return out

    def _note_drawn(self):
        if time.monotonic() - self._drawn_at >= self.IDLE_STOP:
            self._wake.set()    # back in use after a long idle
        self._drawn_at = time.monotonic()

    def draw(self, image, x, y):
        """Draw at (x, y) on a greyscale image, clipped to it.

        While the world is tall this is the top of it, so a row going past
        over the tall world - the overview's, say, or words handed back from
        it - carries the very rows it would have covered, and nothing is lost
        from it for the row having been drawn. Shrinking it back is settle's.
        """
        left, right = max(x, 0), min(x + self.width, image.width)
        if left >= right:
            return
        self._note_drawn()
        pixels = self._over(self._pixels(self._tall or self._small))
        for col in range(left, right):
            for row in range(self.height):
                level = int(pixels[row, col - x])
                if level and 0 <= y + row < image.height:
                    image.putpixel((col, y + row), level)

    def draw_tall(self, image, x, y, cells):
        """Draw the world grown to `cells` tall, a third of a panel row each,
        at (x, y), covering what is under it.

        The first call grows the world the row shows, as it is, so going from
        one to the other is seamless; each call after grows or shrinks it
        further. `cells` can change every frame, and a row only partly grown
        shows dimmed by as much as it is missing.
        """
        self._note_drawn()
        if self._tall is None:
            small = self._small
            self._tall = _View(copy.deepcopy(small.medium), small.clock)
        self._tall.medium.resize(cells)
        pixels = self._over(self._pixels(self._tall))
        image.paste(Image.fromarray(pixels, 'L'), (x, y))

    def settle(self):
        """Shrink the tall world back into the row, keeping what is in the
        top of it, for the row to carry on from. Nothing if it is not tall."""
        if self._tall is None:
            return
        tall, self._tall = self._tall, None
        tall.medium.resize(self.height * SoundMedium.SCALE)
        tall.frame = None
        self._small = tall


class _View:
    """One world and the clock it has been run to."""

    def __init__(self, medium, clock=None):
        self.medium = medium
        self.clock = clock
        self.frame = None           # (time, cells, pixels) last lit, for reuse


_shared = None


def sound_visualizer(width=9):
    """The one visualizer everything shares: one capture, and one object, so a
    line handed between owners can tell it is already showing."""
    global _shared
    if _shared is None:
        _shared = SoundVisualizer(width)
    return _shared
