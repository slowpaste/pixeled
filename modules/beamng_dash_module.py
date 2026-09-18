import logging
import math
import time

import numpy as np
from PIL import Image

from modules.module_base import ModuleBase
from utils.tiny_font import tiny_font
from utils.udp_outgauge_utility import get_telemetry


class _Canvas:
    """A float panel to draw on, 0-255, cleared each frame.

    Every coordinate may be fractional, and every mark is shared between the
    cells it falls across, so a needle can sit between two columns and a shove
    can move the whole dash by a third of a pixel. Marks add and the result is
    clipped once at the end: a dim track under a brighter bar under a needle is
    three passes over the same cells, and each is meant to show through.
    """

    def __init__(self, rows, cols):
        self.buf = np.zeros((rows, cols))
        self.ox = self.oy = 0.0

    def shove(self, ox, oy):
        """Offset everything drawn from here on, in fractions of a pixel."""
        self.ox, self.oy = ox, oy

    def point(self, x, y, level):
        if level <= 0.0:
            return
        x += self.ox
        y += self.oy
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0
        rows, cols = self.buf.shape
        for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            if wy <= 0.0 or not 0 <= yy < rows:
                continue
            for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if wx <= 0.0 or not 0 <= xx < cols:
                    continue
                self.buf[yy, xx] += level * wx * wy

    def blend(self, x, y, level):
        """Light laid *over* what is already there, covering each cell it falls
        across in proportion to how much of it the mark covers. A needle is not
        the sum of itself and the track behind it: at the top of the upshift
        warning that sum is two bright things making one bright thing, and the
        reading disappears into the alarm exactly when it matters."""
        x += self.ox
        y += self.oy
        x0, y0 = math.floor(x), math.floor(y)
        fx, fy = x - x0, y - y0
        rows, cols = self.buf.shape
        for yy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            if wy <= 0.0 or not 0 <= yy < rows:
                continue
            for xx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if wx <= 0.0 or not 0 <= xx < cols:
                    continue
                w = wx * wy
                self.buf[yy, xx] = self.buf[yy, xx] * (1.0 - w) + level * w

    def vline(self, x, y0, y1, level, fade=1.0):
        """A vertical line at a fractional column. `fade` dims every other row,
        which is how the speedometer stays a different object from the
        tachometer without costing it half its pixels."""
        for row in range(y0, y1 + 1):
            self.blend(x, row, level if row % 2 == 0 else level * fade)

    def hline(self, y, x0, x1, level):
        for col in range(x0, x1 + 1):
            self.point(col, y, level)

    def block(self, cx, cy, half, level):
        """A square of side 2*half+1 centred on a fractional point."""
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                self.blend(cx + dx, cy + dy, level)

    def column(self, x0, x1, bottom, filled, level):
        """Bars growing up from `bottom`, `filled` rows tall, the last row of
        them part-lit for the fraction."""
        whole = int(filled)
        for i in range(whole):
            self.hline(bottom - i, x0, x1, level)
        frac = filled - whole
        if frac > 0.0:
            self.hline(bottom - whole, x0, x1, level * frac)

    def image(self):
        np.clip(self.buf, 0.0, 255.0, out=self.buf)
        return Image.fromarray(np.rint(self.buf).astype(np.uint8), 'L')


class BeamngDashModule(ModuleBase):
    """The whole 9x34 panel as one BeamNG dashboard.

    Drawn in greyscale on a float canvas, so nothing here is on or off. Needles
    sit between columns and are shared across the two they fall between, guides
    sit far below the readings they carry rather than being dithered away from
    them, and the whole dash is shoved around by the car's own acceleration.

    Top to bottom, with a blank row between each section:

        rows  0- 4   tachometer, a needle over a dim bar, redline zone marked
        rows  6- 9   speedometer, the same needle with alternate rows dimmed
        rows 11-21   6-speed H pattern, dim gate, bright marker with a trail
        rows 23-33   clutch, brake and throttle, three bars over dim tracks

    Brightness is what separates a guide from a reading. The gate, the sweep
    tracks and the pedals' empty travel sit at a tenth or so of full, the bar
    filling each sweep at a fifth, and only the needles, the shift marker and
    the pedals themselves are fully lit. On a panel nine pixels wide that is
    the difference between a dash that reads at a glance and a lit box with
    brighter spots in it.

    Both needles are placed to a fraction of a column. A 9-wide sweep gives 8
    pixels of travel, and sharing a needle's light between the pair it falls
    between makes the distance between two of them legible - an idle creeping
    up, a speed settling - where whole-pixel steps gave 9 readings and a
    flicker between them.

    The upshift warning breathes rather than blinks. The redline zone brightens
    and falls back a few times a second while the needle stays fully lit above
    it, so the alarm is unmistakable and the reading is never interrupted. The
    old 1-bit dash inverted the whole band, which was the only thing a panel
    with one brightness could do.

    Full scale for the tachometer is learned from the highest rpm this vehicle
    has been seen to reach, the same trick the sand gauge uses for watts, since
    OutGauge never says where the redline is. Nothing is persisted: one pull to
    the limiter relearns it, which is cheaper than another runtime state file
    for deploy.sh to protect.

    Knowing when to throw that away is the awkward part. BeamNG sends "beam" as
    the car name for every vehicle, so there is no identity in the packet to
    watch, and a scale learned from a 9000rpm engine would otherwise put the
    upshift point of the next car out of its reach entirely - the tachometer
    would just quietly stop flashing. The signal used instead is a break in the
    stream: swapping vehicles or loading a map stops the packets for seconds,
    where driving never does. A pause menu resets it too, which costs one pull
    to relearn and is the harmless direction to be wrong in.
    """

    # Section heights, top to bottom. The pedal bars take whatever is left, so
    # only these three plus the gaps are fixed.
    TACH_ROWS = 5
    SPEED_ROWS = 4
    H_ROWS = 11
    MIN_PEDAL_ROWS = 4
    GAP = 1

    PEDALS = ('clutch', 'brake', 'throttle')   # left to right, as in the footwell

    # Brightness. Guides an order below readings; see the class docstring.
    TRACK = 16          # the sweep a needle has to travel
    BAR = 44            # zero up to the needle, filled in
    REDLINE = 70        # the share of the tachometer past the upshift point
    WARN_PEAK = 165     # what that rises to at the top of the warning's breath
    WARN_WASH = 40      # and what the rest of the band rises to with it
    GATE = 26           # the H pattern the shift marker runs on
    NODE = 72           # where a gear sits on it
    NEEDLE = 255
    BG_CEILING = 150    # the most any guide may reach, so a needle always tells
    MARKER = 255
    TRAIL = 0.5         # share of full the marker's wake carries
    TRAIL_TAU = 0.11    # seconds it takes to fade by e
    PEDAL = 255
    PEDAL_TRACK = 14    # the travel a pedal has left

    MAX_DT = 0.25          # clamp, so a stall can't teleport the shift marker
    DL_SHIFT = 0x01        # OutGauge dash light bit for the shift indicator
    RPM_CEILING = 20000.0  # above this it is a bad packet, not a real engine
    SPEED_CEILING = 200.0  # m/s
    # Seconds without a packet before the dash falls back to its resting face.
    # OutGauge streams at tens of packets a second, so this only trips on a
    # pause, a menu, or the game going away.
    STALE_AFTER = 1.5
    # A gap this long is a vehicle swap or a map load, not a dropped packet, so
    # the learned scales are thrown away and relearned for whatever comes back.
    RESET_AFTER = 5.0

    G = 9.80665
    # The dash hangs on a spring: 1.9Hz, and damped to about a third, so a hit
    # rings twice and is gone rather than either snapping back or wobbling on.
    SPRING = 140.0
    DAMPING = 8.0
    LEAN_TAU = 0.10        # seconds of smoothing on the acceleration behind the lean
    PACKET_MIN = 0.002     # bounds on the gap between two packets that a
    PACKET_MAX = 0.5       # speed difference may be divided by
    FLASH_LEVEL = 85       # how much light a collision throws over everything
    FLASH_TAU = 0.16       # seconds that takes to fade by e
    # Longest step either spring is integrated over. Both are stiff enough to
    # come apart if a whole frame is handed to them at once: at stock firmware's
    # ~6fps a frame is longer than the marker's wobble, and the dash's own
    # spring is only just inside stability. Frames are cut into steps this long
    # instead, which costs a handful of multiplications on a slow frame and
    # nothing at all on a fast one.
    SPRING_STEP = 0.01
    SETTLE_KICK = 7.0      # px/s the marker carries into the gate it arrives in
    SETTLE_SPRING = 700.0
    SETTLE_DAMPING = 26.0

    def __init__(self, height=34, max_speed=55.0, redline_floor=4500.0,
                 shift_fraction=0.92, shift_release=0.05, shift_speed=30.0,
                 neutral_return=0.8, flash_hz=8.0, motion=True, lean_px=1.5,
                 lean_g=1.1, impact_g=5.0, shake_px=2.5):
        """
        :param max_speed: m/s at the right edge of the speedometer. Only a
            starting point - the scale grows if it is ever exceeded, so the
            needle never sits pinned and silent.
        :param redline_floor: rpm the tachometer assumes as full scale until it
            has watched the engine rev higher. Keeps a car that has only idled
            from scaling 900rpm across the whole panel.
        :param shift_fraction: share of the learned redline that starts the
            tachometer breathing, and where its marked zone begins.
        :param shift_release: extra share it has to drop back through before the
            warning stops, so a needle sitting on the threshold does not stutter
            in and out of it.
        :param shift_speed: pixels/sec the marker travels along the H pattern.
            30 puts a 1-2 shift at roughly a third of a second.
        :param neutral_return: seconds the lever has to sit in neutral before it
            counts as parked there rather than passing through, and slides back
            to the centre of the gate plane. Longer than any shift, shorter than
            any deliberate stop in neutral.
        :param flash_hz: breaths per second of the upshift warning. Capped at
            whatever the measured frame rate can actually resolve, so the same
            setting reads as a pulse rather than as an aliased stutter on stock
            firmware, which draws at a tenth of the patched firmware's rate.
        :param motion: whether the car's acceleration moves the dash at all.
        :param lean_px: pixels the dash leans at lean_g.
        :param lean_g: acceleration, in g, that leans it that far.
        :param impact_g: acceleration no tyre can produce, so anything past it
            is taken for a collision: it kicks the dash sideways as well as
            along, and throws a brief wash of light over the whole panel.
        :param shake_px: the furthest the dash is ever allowed to be shoved,
            so a heavy crash cannot push a gauge off the panel.
        """
        super().__init__(height)
        # Both are divisors every frame, so a zero from config.json would take
        # the panel out rather than just mis-scale it.
        self.max_speed = max(max_speed, 1.0)
        self.redline_floor = max(redline_floor, 1.0)
        self.shift_fraction = shift_fraction
        self.shift_release = shift_release
        self.shift_speed = shift_speed
        self.neutral_return = neutral_return
        self.flash_hz = flash_hz
        self.motion = motion
        self.lean_px = lean_px
        self.lean_g = max(lean_g, 0.1)
        self.impact_g = impact_g
        self.shake_px = shake_px

        self.tach_y = 0
        self.speed_y = self.tach_y + self.TACH_ROWS + self.GAP
        self.h_y = self.speed_y + self.SPEED_ROWS + self.GAP
        self.pedal_y = self.h_y + self.H_ROWS + self.GAP
        self.pedal_rows = height - self.pedal_y
        if self.pedal_rows < self.MIN_PEDAL_ROWS:
            raise ValueError(
                f"BeamngDashModule needs at least "
                f"{self.pedal_y + self.MIN_PEDAL_ROWS} rows, got {height}")

        # H pattern rows: the gear ends sit one row in from each edge of the
        # section so the 3x3 marker still fits when parked on them.
        self.top_y = self.h_y + 1
        self.bot_y = self.h_y + self.H_ROWS - 2
        self.rail_y = (self.top_y + self.bot_y) // 2

        # Column geometry depends on the width, which only arrives at render.
        self.gates = None
        self.nodes = {}
        self._wake = None        # the marker's fading trail, panel-sized

        self._marker = None      # [x, y] in panel coords, fractional in flight
        self._goal = None
        self._route = []
        self._gate_x = None      # gate the last engaged gear was in
        self._neutral_dwell = 0.0
        self._last_render = None
        self._settle = 0.0       # px of overshoot left in the gate it arrived in
        self._settle_v = 0.0
        self._settle_dir = (0.0, 1.0)

        self._car = None
        self._redline = self.redline_floor
        self._speed_scale = self.max_speed
        self._warning = False      # past the upshift point, tachometer breathing
        self._warning_since = 0.0  # so the breath always starts from dark
        self._dt_ema = None        # measured frame interval, for the pulse cap

        self._shove = [0.0, 0.0]   # px the whole dash is displaced by
        self._shove_v = [0.0, 0.0]
        self._accel = 0.0          # m/s^2 along the car, smoothed
        self._flash = 0.0          # 0-1, light thrown by a collision
        self._impacts = 0
        self._last_packet_at = None
        self._last_speed = 0.0

        self._live = False
        self._last_live_at = None
        self._seen_gears = set()   # for the log trail, see _note()

    # ── geometry ─────────────────────────────────────────────────────────────
    def _ensure_geometry(self, width):
        """Place the three gates and the gear each one holds.

        At the panel's 9 columns the gates land on 1, 4 and 7, which is exactly
        three 3-wide marker slots side by side with nothing left over.
        """
        if self.gates is not None:
            return
        self.gates = (1, width // 2, width - 2)
        left, mid, right = self.gates
        # OutGauge numbers gears from reverse: 0 is R, 1 is neutral, 2 is first.
        self.nodes = {
            1: (mid, self.rail_y),
            2: (left, self.top_y),   3: (left, self.bot_y),
            4: (mid, self.top_y),    5: (mid, self.bot_y),
            6: (right, self.top_y),  7: (right, self.bot_y),
        }
        self._marker = list(self.nodes[1])
        self._goal = self.nodes[1]
        self._gate_x = mid
        self._wake = np.zeros((self.height, width))

    def _node_for(self, gear):
        """Where an engaged forward gear parks the marker. Reverse and neutral
        are decided in _update_marker and never reach this.

        Seventh and eighth in the odd car that has them share sixth's slot: a
        6-speed pattern has nowhere else to put them, and the alternative is a
        marker that vanishes off the top of the box.
        """
        return self.nodes.get(gear, self.nodes[7])

    # ── state ────────────────────────────────────────────────────────────────
    def _advance(self):
        """Seconds since the last frame, clamped so a stall can't make the
        first frame after it jump."""
        now = time.monotonic()
        last, self._last_render = self._last_render, now
        dt = 0.0 if last is None else min(now - last, self.MAX_DT)
        if dt > 0.0:
            self._dt_ema = dt if self._dt_ema is None else 0.9 * self._dt_ema + 0.1 * dt
        return dt

    def _reset_scales(self, why):
        self._redline = self.redline_floor
        self._speed_scale = self.max_speed
        logging.info("BeamngDash: relearning full scale (%s)", why)

    def _track_scales(self, tel):
        """Grow the learned full scales.

        The car name check is kept even though BeamNG pins it to "beam": it
        costs nothing and is the right hook the day a packet carries a real one.
        """
        car = tel.get('car_name', '')
        if self._car is not None and car != self._car:
            self._reset_scales(f"car name {self._car!r} -> {car!r}")
        self._car = car
        rpm = tel.get('rpm', 0.0)
        if self._redline < rpm < self.RPM_CEILING:
            self._redline = rpm
        speed = tel.get('speed', 0.0)
        if self._speed_scale < speed < self.SPEED_CEILING:
            self._speed_scale = speed

    def _note(self, tel, live, gear):
        """Log the feed coming and going, and every distinct gear value seen.

        A gearbox produces a handful of these a session, so the volume is
        nothing, and it is the one thing that can't be worked out from the
        panel: if the game ever numbers its gears differently the log says so
        outright, instead of leaving a marker that mysteriously never moves.
        """
        if live != self._live:
            self._live = live
            if live:
                logging.info("BeamngDash: telemetry live, car=%r gear=%d rpm=%.0f "
                             "time_ms=%d", tel.get('car_name'), gear,
                             tel.get('rpm', 0.0), tel.get('time_ms', 0))
            else:
                logging.info("BeamngDash: telemetry stale, resting at neutral")
        if live and gear not in self._seen_gears:
            self._seen_gears.add(gear)
            logging.info("BeamngDash: gear value %d reads as %s", gear,
                         self._gear_label(gear))

    def _gear_label(self, gear):
        if gear <= 0:
            return 'reverse'
        if gear == 1:
            return 'neutral'
        if gear <= 7:
            return f'{gear - 1}'
        return f'{gear - 1}, clamped onto 6th'

    # ── motion ───────────────────────────────────────────────────────────────
    def _update_motion(self, dt, tel, live):
        """Shove the whole dash about from the car's own acceleration.

        OutGauge carries no accelerometer: no lateral g, no vertical, no
        heading, not even a steering angle. The one motion in the packet is the
        speed, so the one force that can be recovered is the one along the car,
        differentiated between packets. Braking leans the dash down the panel
        and accelerating lifts it, the way the car's own mass moves.

        Anything past impact_g is not something tyres can do to a car, so it is
        taken for a collision rather than for driving: it kicks the spring
        sideways as well, which a lean never does, and throws a wash of light
        over the panel that fades in a fifth of a second. The sideways kick
        alternates, because nothing in the packet says which side was hit.

        The dash hangs on a spring either way, so the lean arrives with a little
        overshoot and a hit rings out instead of snapping back.
        """
        if not self.motion:
            self._shove = [0.0, 0.0]
            self._flash = 0.0
            return

        received = tel.get('received_at') if live else None
        if received is not None and received != self._last_packet_at:
            gap = None if self._last_packet_at is None else received - self._last_packet_at
            speed = tel.get('speed', 0.0)
            if gap is not None and self.PACKET_MIN <= gap <= self.PACKET_MAX:
                accel = (speed - self._last_speed) / gap
                if abs(accel) > self.impact_g * self.G:
                    self._impacts += 1
                    hit = min(abs(accel) / (self.impact_g * self.G), 4.0)
                    self._shove_v[1] += math.copysign(9.0 * hit, -accel)
                    self._shove_v[0] += 6.0 * hit * (1 if self._impacts % 2 else -1)
                    self._flash = min(1.0, self._flash + 0.5 * hit)
                    logging.info("BeamngDash: %.0f m/s^2 in %.0fms reads as a hit",
                                 accel, gap * 1000)
                    # A collision is not a reading about how hard it is braking.
                    accel = 0.0
                weight = 1.0 - math.exp(-gap / self.LEAN_TAU)
                self._accel += (accel - self._accel) * weight
            self._last_packet_at = received
            self._last_speed = speed
        elif not live:
            self._accel += (0.0 - self._accel) * min(1.0, dt / self.LEAN_TAU)

        # Braking is a negative acceleration and leans the dash down the panel,
        # so the target takes the sign as it comes.
        lean = -self._accel / (self.lean_g * self.G)
        target = (0.0, max(-1.0, min(1.0, lean)) * self.lean_px)
        for step in self._steps(dt):
            for axis in (0, 1):
                pull = -self.SPRING * (self._shove[axis] - target[axis])
                self._shove_v[axis] += (pull - self.DAMPING * self._shove_v[axis]) * step
                self._shove[axis] += self._shove_v[axis] * step
                if abs(self._shove[axis]) > self.shake_px:
                    self._shove[axis] = math.copysign(self.shake_px, self._shove[axis])
                    self._shove_v[axis] = 0.0
        if self._flash > 0.0:
            self._flash *= math.exp(-dt / self.FLASH_TAU) if dt else 1.0
            if self._flash < 0.004:
                self._flash = 0.0

    def _steps(self, dt):
        """One frame as a sequence of steps short enough to integrate over."""
        if dt <= 0.0:
            return ()
        count = max(1, int(math.ceil(dt / self.SPRING_STEP)))
        return (dt / count,) * count

    def _plan(self, target):
        """Waypoints from wherever the marker is to a gear, the way a hand moves
        one: out to the neutral rail, across it, then into the gate.

        A shift interrupted mid-flight replans from the marker's current spot,
        so the double-shift a driver actually makes reads as one continuous
        sweep rather than a stutter through the abandoned gear.
        """
        x, y = self._marker
        tx, ty = target
        if abs(x - tx) < 1e-6:
            return [(tx, ty)]                       # same gate, straight through
        return [(x, self.rail_y), (tx, self.rail_y), (tx, ty)]

    def _neutral_target(self, dt):
        """Where neutral puts the marker.

        A shift passes through the neutral of the gate it is leaving, not
        through the middle of the gate plane: 1-2 is one straight pull down the
        left gate, and sending it via the centre draws a dogleg the lever never
        makes. Only once it has been left there longer than any shift takes does
        it count as parked, and slide back to the centre the way a sprung lever
        does.
        """
        self._neutral_dwell += dt
        if self._neutral_dwell >= self.neutral_return:
            return self.nodes[1]
        return (self._gate_x, self.rail_y)

    def _update_marker(self, dt, gear):
        if gear <= 0:
            # Reverse draws its own glyph, so hold the marker at the centre of
            # the rail: that is where the lever passes through on the way out.
            self._marker = list(self.nodes[1])
            self._goal = self.nodes[1]
            self._route = []
            self._gate_x = self.nodes[1][0]
            self._neutral_dwell = 0.0
            return
        if gear == 1:
            node = self._neutral_target(dt)
        else:
            node = self._node_for(gear)
            self._gate_x = node[0]
            self._neutral_dwell = 0.0
        if node != self._goal:
            self._goal = node
            self._route = self._plan(node)

        budget = self.shift_speed * dt
        while self._route and budget > 0:
            tx, ty = self._route[0]
            dx, dy = tx - self._marker[0], ty - self._marker[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= budget:
                self._marker = [tx, ty]
                budget -= dist
                self._route.pop(0)
                if not self._route and dist > 1e-6:
                    # Into the gate and against its stop: the lever arrives
                    # with some speed left and rattles it off in a wobble too
                    # small to see one frame at a time, and unmistakable at 60.
                    self._settle_dir = (dx / dist, dy / dist)
                    self._settle_v += self.SETTLE_KICK
            else:
                self._marker[0] += dx / dist * budget
                self._marker[1] += dy / dist * budget
                budget = 0.0

        for step in self._steps(dt):
            pull = (-self.SETTLE_SPRING * self._settle
                    - self.SETTLE_DAMPING * self._settle_v)
            self._settle_v += pull * step
            self._settle += self._settle_v * step

    def _shift_up(self, tel, gear, live, now):
        """Whether it is time to upshift, and so whether the tachometer warns.

        Neutral and reverse never warn: revving out a disengaged engine is not a
        cue to shift. The game's own shift light wins when the car has one -
        dashLights says the light exists, showLights says it is on - and the
        learned redline covers every car that doesn't.
        """
        was = self._warning
        if not live or gear < 2:
            self._warning = False
        else:
            lights = int(tel.get('dashLights', 0)) & int(tel.get('showLights', 0))
            if lights & self.DL_SHIFT:
                self._warning = True
            else:
                fraction = self.shift_fraction - (self.shift_release if was else 0.0)
                self._warning = tel.get('rpm', 0.0) >= self._redline * fraction
        if self._warning and not was:
            self._warning_since = now
        return self._warning

    def _warn_pulse(self, now):
        """How far into the warning's breath we are, 0 dark to 1 brightest.

        Capped so a breath always spans at least four frames: asking for 8Hz at
        stock firmware's ~6fps samples the cycle at its own frequency and comes
        out as an erratic stutter rather than as a pulse. A cosine from zero
        means the warning always starts by getting brighter.
        """
        hz = self.flash_hz
        if self._dt_ema:
            hz = min(hz, 1.0 / (4.0 * self._dt_ema))
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * hz * (now - self._warning_since))

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw_sweep(self, canvas, width, y0, rows, value, fade=1.0, zone=None,
                    wash=0.0):
        """One gauge: a dim track the full width, a brighter bar from zero up to
        the reading, and the needle itself at the fractional column it falls on.

        The bar is what makes a glance work - how far along the sweep is, read
        as an area rather than as the position of one line - and the needle is
        what makes a small change visible, since it moves between columns
        instead of jumping from one to the next.
        """
        value = min(max(value, 0.0), 1.0)
        span = width - 1
        x = value * span
        for row in range(y0, y0 + rows):
            dim = 1.0 if row % 2 == 0 else fade
            self._track_row(canvas, width, row, (self.TRACK + wash) * dim)
            if zone is not None:
                start, zone_level = zone
                for col in range(int(math.ceil(start * span)), width):
                    canvas.point(col, row, zone_level * dim)
            whole = int(x)
            for col in range(whole):
                canvas.point(col, row, self.BAR * dim)
            edge = x - whole
            if edge > 0.0:
                canvas.point(whole, row, self.BAR * dim * edge)
        # Everything so far is guide: hold it below the needle before the
        # needle goes over it, so the reading survives its own warning.
        band = canvas.buf[y0:y0 + rows]
        np.clip(band, 0.0, self.BG_CEILING, out=band)
        canvas.vline(x, y0, y0 + rows - 1, self.NEEDLE, fade)

    @staticmethod
    def _track_row(canvas, width, row, level):
        for col in range(width):
            canvas.point(col, row, level)

    def _draw_shifter(self, canvas, dt):
        """The H pattern, plus the marker running on it.

        The gate is drawn as solid dim lines rather than dithered to every other
        pixel. Dither was how a 1-bit panel said "this is a guide, not a
        reading"; with 256 levels the same thing is said by brightness, and the
        gate comes out as one continuous shape instead of a dotted suggestion of
        one. Every gear position is marked a little brighter still, so the six
        ends and the neutral centre can be counted without the marker on them.

        The marker leaves a wake, kept in a buffer that fades by e every
        TRAIL_TAU. At 60fps a shift reads as a stroke drawn by something moving,
        which is most of what tells a 1-2 from a 3-4 at this size.
        """
        left, _, right = self.gates
        for gx in self.gates:
            for y in range(self.top_y, self.bot_y + 1):
                canvas.point(gx, y, self.GATE)
        canvas.hline(self.rail_y, left, right, self.GATE)
        for node in self.nodes.values():
            canvas.point(node[0], node[1], self.NODE - self.GATE)

        mx = self._marker[0] + self._settle_dir[0] * self._settle
        my = self._marker[1] + self._settle_dir[1] * self._settle

        wake = _Canvas(*self._wake.shape)
        wake.shove(canvas.ox, canvas.oy)
        wake.block(mx, my, 1, self.MARKER)
        if dt > 0.0:
            self._wake *= math.exp(-dt / self.TRAIL_TAU)
        np.maximum(self._wake, wake.buf, out=self._wake)
        canvas.buf += self._wake * self.TRAIL
        canvas.block(mx, my, 1, self.MARKER)

    def _draw_reverse(self, canvas, width):
        """Reverse gets a letter, not a marker position.

        Three gates fill the width, and every one of their six ends is a forward
        gear, so there is no free slot to park reverse in. A scaled-up R fills
        the same box instead and can't be mistaken for a gear. The gate stays
        dimly behind it, so the section keeps its shape while the lever is out
        of the pattern altogether.
        """
        for gx in self.gates:
            for y in range(self.top_y, self.bot_y + 1):
                canvas.point(gx, y, self.GATE)
        canvas.hline(self.rail_y, self.gates[0], self.gates[-1], self.GATE)

        glyph = tiny_font['R']
        scale = max(1, min(self.H_ROWS // len(glyph), width // len(glyph[0])))
        x0 = (width - len(glyph[0]) * scale) // 2
        y0 = self.h_y + (self.H_ROWS - len(glyph) * scale) // 2
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit != '1':
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        canvas.point(x0 + col * scale + dx,
                                     y0 + row * scale + dy, self.MARKER)

    def _draw_pedals(self, canvas, width, tel):
        """Clutch, brake and throttle, growing up from the bottom edge.

        Three bars of a third of the width each, with no gap between them: at
        nine columns a separator would cost a third of every bar. Instead each
        pedal's empty travel is a dim line up the middle of its own bar, so the
        three read as three even when none of them is touched - a dim track the
        full width of each would have merged into one unbroken block, which is
        the same thing as drawing nothing.

        The top row of each bar is lit for the fraction of a pixel it covers, so
        a pedal eased on moves smoothly instead of in eleven steps, and two
        neighbours at the same height still merge into one block, which is a
        fair reading of two pedals at the same travel.
        """
        bar_w = max(1, width // len(self.PEDALS))
        bottom = self.height - 1
        for i, pedal in enumerate(self.PEDALS):
            x0 = i * bar_w
            x1 = x0 + bar_w - 1
            for row in range(self.pedal_y, bottom + 1):
                canvas.point((x0 + x1) / 2.0, row, self.PEDAL_TRACK)
            value = min(max(tel.get(pedal, 0.0), 0.0), 1.0)
            canvas.column(x0, x1, bottom, value * self.pedal_rows, self.PEDAL)

    def render(self, width):
        self._ensure_geometry(width)
        dt = self._advance()

        tel = get_telemetry()
        # Liveness comes from when the last packet landed, never from the packet
        # contents: the reader's defaults are all legitimate readings at rest,
        # and its zeroed gear would otherwise leave an idle panel claiming
        # reverse. Everything downstream then works off one dict, so a stale
        # feed rests the whole dash rather than freezing half of it.
        now = time.monotonic()
        received = tel.get('received_at')
        live = received is not None and now - received < self.STALE_AFTER
        if live:
            if self._last_live_at is not None and now - self._last_live_at > self.RESET_AFTER:
                self._reset_scales(f"{now - self._last_live_at:.0f}s gap in the feed")
            self._last_live_at = now
            self._track_scales(tel)
        else:
            tel = {}
        gear = int(tel.get('gear', 1))
        self._note(tel, live, gear)
        self._update_motion(dt, tel, live)
        self._update_marker(dt, gear)

        canvas = _Canvas(self.height, width)
        canvas.shove(self._shove[0], self._shove[1])

        warning = self._shift_up(tel, gear, live, now)
        pulse = self._warn_pulse(now) if warning else 0.0
        zone = (self.shift_fraction,
                self.REDLINE + pulse * (self.WARN_PEAK - self.REDLINE))
        # The rest of the band comes up with the zone, so the warning is a whole
        # block breathing rather than a bright edge appearing - and both are
        # held under BG_CEILING, so the needle stays the brightest thing in the
        # row at every point in the breath.
        self._draw_sweep(canvas, width, self.tach_y, self.TACH_ROWS,
                         tel.get('rpm', 0.0) / self._redline, zone=zone,
                         wash=pulse * self.WARN_WASH)
        self._draw_sweep(canvas, width, self.speed_y, self.SPEED_ROWS,
                         tel.get('speed', 0.0) / self._speed_scale, fade=0.45)
        if gear <= 0:
            self._draw_reverse(canvas, width)
        else:
            self._draw_shifter(canvas, dt)
        self._draw_pedals(canvas, width, tel)

        if self._flash > 0.0:
            canvas.buf += self._flash * self.FLASH_LEVEL
        return canvas.image()
