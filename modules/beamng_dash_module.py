import logging
import math
import time

import numpy as np
from PIL import Image

from modules.module_base import ModuleBase
from utils.plug_physics import Ripple
from utils.tiny_font import tiny_font
from utils.udp_outgauge_utility import get_telemetry
from utils.udp_motionsim_utility import get_motion, motionsim_reader


class _Scene:
    """A float panel to draw on, 0-255, with depth: the dash as a small solid
    thing held in front of the panel, seen from a little way off.

    Everything is placed in the panel's own terms - x across in columns, y
    down in rows, both counted from pixel centres - plus z, pixels up out of
    the panel toward whoever is looking at it. The whole scene is carried by
    one rigid pose, turned about the middle of the panel and then moved, and
    is seen in perspective from EYE pixels above that middle: what is raised
    comes out a little larger and further from the centre, and what sinks
    goes a little smaller and toward it. Tipped, a thing standing up off the
    panel swings across what lies flat under it, which is the whole point of
    giving the dash depth on a panel nine pixels wide.

    Every mark is a sample with a weight, shared between the four cells it
    lands between, so nothing is on or off and nothing snaps to a pixel.
    Lines and areas are sampled densely and weighted by how much length or
    area each sample stands for, so a line is as bright wherever it runs and
    at whatever angle the pose has put it. Marks either add - a dim track
    under a brighter zone, each meant to show through - or are laid over what
    is there, covering each cell in proportion to how much of it they cover:
    a needle is not the sum of itself and the track behind it.

    A mark drawn with shade=True is lit by how high it stands, as a fake
    ambient occlusion: whatever is tucked down into the panel gets less of the
    light and whatever stands up out of it gets more, by e every SHADE_DEPTH
    pixels. It is the height after the pose that counts, so a gate lying flat
    on the panel darkens along its low side and brightens along its high one
    as the dash rolls.
    """

    EYE = 40.0          # pixels from the eye to the panel
    SHADE_DEPTH = 2.5   # pixels of height that light or darken a mark by e

    def __init__(self, rows, cols):
        self.buf = np.zeros((rows, cols))
        self.rows, self.cols = rows, cols
        self.cx, self.cy = (cols - 1) / 2.0, (rows - 1) / 2.0
        self.turn = np.eye(3)
        self.move = np.zeros(3)
        self._over_w = np.zeros((rows, cols))
        self._over_v = np.zeros((rows, cols))

    def pose(self, roll=0.0, pitch=0.0, yaw=0.0, move=(0.0, 0.0, 0.0)):
        """Tip the scene about the middle of the panel, then move it.

        Roll is about the panel's long axis, positive taking the right side
        down into it; pitch about its short axis, positive bringing the top
        up out of it; yaw about the axis out of it, positive clockwise as
        seen. Radians; `move` is in pixels, x, y and z.
        """
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        roll_m = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
        pitch_m = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
        yaw_m = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        self.turn = yaw_m @ pitch_m @ roll_m
        self.move = np.asarray(move, dtype=float)

    def place(self, pts):
        """Scene points, (N, 3), where the pose puts them."""
        pivot = np.array([self.cx, self.cy, 0.0])
        return (pts - pivot) @ self.turn.T + pivot + self.move

    def project(self, pts):
        """Scene points, (N, 3), to where they land on the panel, (N, 2)."""
        placed = self.place(pts)
        scale = self.EYE / np.maximum(self.EYE - placed[:, 2], 1.0)
        xs = self.cx + (placed[:, 0] - self.cx) * scale
        ys = self.cy + (placed[:, 1] - self.cy) * scale
        return xs, ys

    def shade(self, z):
        """How much light a mark at height `z`, as placed, gets."""
        return np.exp(np.asarray(z, dtype=float) / self.SHADE_DEPTH)

    def _splat(self, pts, level, weight, over, shade=False):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        xs, ys = self.project(pts)
        level = np.broadcast_to(np.asarray(level, dtype=float), xs.shape)
        if shade:
            level = np.minimum(level * self.shade(self.place(pts)[:, 2]), 255.0)
        weight = np.broadcast_to(np.asarray(weight, dtype=float), xs.shape)
        x0, y0 = np.floor(xs), np.floor(ys)
        fx, fy = xs - x0, ys - y0
        x0, y0 = x0.astype(int), y0.astype(int)
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            for dy, wy in ((0, 1.0 - fy), (1, fy)):
                xx, yy = x0 + dx, y0 + dy
                w = wx * wy * weight
                ok = (xx >= 0) & (xx < self.cols) & (yy >= 0) & (yy < self.rows) & (w > 0)
                if not ok.any():
                    continue
                at = (yy[ok], xx[ok])
                if over:
                    np.add.at(self._over_w, at, w[ok])
                    np.add.at(self._over_v, at, w[ok] * level[ok])
                else:
                    np.add.at(self.buf, at, w[ok] * level[ok])

    def point(self, p, level, over=False, shade=False):
        self._splat([p], level, 1.0, over, shade)

    def line(self, a, b, level, over=False, step=0.25, shade=False):
        """A line a pixel wide from `a` to `b`, `level` all along it."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / step)))
        t = (np.arange(n) + 0.5) / n
        self._splat(a + np.outer(t, b - a), level, length / n, over, shade)

    def curve(self, pts, levels, over=False):
        """A line through `pts`, (N, 3), each stretch carrying its own level:
        `levels` has one fewer entry than there are points."""
        pts = np.asarray(pts, dtype=float)
        mids = (pts[1:] + pts[:-1]) / 2.0
        lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        self._splat(mids, levels, lengths, over)

    # Samples across each pixel, each way, that a filled shape is measured by.
    COVER = 5

    def quad(self, corner, u, v, level, over=False, shade=False):
        """The parallelogram from `corner` along `u` and `v`, filled.

        Filled by how much of each pixel it covers, not by sharing samples
        out between neighbours the way a line is: shared out, a square of
        pixels comes out as a blur a pixel wider all round, and one sitting
        squarely on its pixels at rest should look it.
        """
        corner, u, v = (np.asarray(q, dtype=float) for q in (corner, u, v))
        xs, ys = self.project(np.array([corner, corner + u, corner + u + v, corner + v]))
        x0 = max(int(math.floor(xs.min() + 0.5)), 0)
        x1 = min(int(math.ceil(xs.max() - 0.5)), self.cols - 1)
        y0 = max(int(math.floor(ys.min() + 0.5)), 0)
        y1 = min(int(math.ceil(ys.max() - 0.5)), self.rows - 1)
        if x1 < x0 or y1 < y0:
            return
        n = self.COVER
        sub = (np.arange(n) + 0.5) / n - 0.5
        px = (np.arange(x0, x1 + 1)[:, None] + sub).ravel()
        py = (np.arange(y0, y1 + 1)[:, None] + sub).ravel()
        gx, gy = np.meshgrid(px, py)
        # Inside a convex outline is on the same side of every edge of it,
        # whichever way round the pose has left the corners.
        sides = []
        for i in range(4):
            ax, ay, bx, by = xs[i], ys[i], xs[(i + 1) % 4], ys[(i + 1) % 4]
            sides.append((bx - ax) * (gy - ay) - (by - ay) * (gx - ax))
        sides = np.array(sides)
        inside = (sides >= 0).all(axis=0) | (sides <= 0).all(axis=0)
        rows, cols = y1 - y0 + 1, x1 - x0 + 1
        lit = inside * float(level)
        if shade:
            # The height under each sample, where the line from the eye
            # through it meets the quad's plane: a tipped or hinged face is
            # lit a little differently all the way across.
            p0, p1, _, p3 = self.place(np.array([corner, corner + u, corner + u + v,
                                                 corner + v]))
            normal = np.cross(p1 - p0, p3 - p0)
            eye = np.array([self.cx, self.cy, self.EYE])
            ray_n = (normal[0] * (gx - self.cx) + normal[1] * (gy - self.cy)
                     - normal[2] * self.EYE)
            ray_n = np.where(np.abs(ray_n) < 1e-9, 1e-9, ray_n)
            t = float(normal @ (p0 - eye)) / ray_n
            lit = np.minimum(lit * self.shade(self.EYE * (1.0 - t)), 255.0)
        cover = inside.reshape(rows, n, cols, n).mean(axis=(1, 3))
        light = lit.reshape(rows, n, cols, n).mean(axis=(1, 3))
        cells = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        if over:
            self._over_w[cells] += cover
            self._over_v[cells] += light
        else:
            self.buf[cells] += light

    def lay_over(self):
        """Put everything drawn with over=True on top of the rest."""
        cover = np.minimum(self._over_w, 1.0)
        level = self._over_v / np.maximum(self._over_w, 1e-9)
        self.buf = self.buf * (1.0 - cover) + level * cover
        self._over_w[:] = 0.0
        self._over_v[:] = 0.0

    def at_rest(self, x, y, z):
        """The scene point at height `z` that, with the dash at rest, is
        seen at (x, y) on the panel. Something raised is seen a little out
        from the middle, and placed with this it still sits squarely on the
        pixels it is meant to at rest, and only leaves them when the dash
        moves."""
        shrink = (self.EYE - z) / self.EYE
        return (self.cx + (x - self.cx) * shrink,
                self.cy + (y - self.cy) * shrink, z)

    def fresh(self):
        """An empty scene under the same pose, for drawing one thing alone."""
        other = _Scene(self.rows, self.cols)
        other.turn, other.move = self.turn, self.move
        return other

    def image(self):
        np.clip(self.buf, 0.0, 255.0, out=self.buf)
        return np.rint(self.buf).astype(np.uint8)


class BeamngDashModule(ModuleBase):
    """The whole 9x34 panel as one BeamNG dashboard.

    Drawn in greyscale into a small 3D scene, so nothing here is on or off and
    nothing sits flat for want of a way not to. Top to bottom:

        rows  0- 6   tachometer: a dial tilted back, its needle's arm swept
                     round the arc
        rows  7-13   speedometer: the same dial, the same tilt, tucked up
                     against the first where the tilt leaves room
        rows 16-24   6-speed H pattern: a dim gate, and the gear knob held
                     above it on a lever, trailing a wake
        rows 28-32   clutch, brake and throttle: three lit faces, pressed
                     down into the panel's shadow with the pedal, over a
                     dark bottom row for a pressed pedal's foot to be seen
                     drawing away from

    The two dials are one shape, a circular arc of most of a turn about a
    pivot in the middle of the panel's width, and both are tipped back out of
    the panel the way a binnacle faces the driver: seen from straight on the
    arc comes out flattened, and as the car rolls or pitches the raised top
    of it swings across the panel against the rest. The needle is drawn as
    its outer arm rather than as a tip, bright over a thinner, dimmer arc, so
    what is a reading and what is a guide is never in doubt.

    The knob stands above its gate and the pedals sink below theirs, so the
    dash reads as layers when it moves: tip it and the knob slides over the
    gate on its lever, press a pedal and it drops away into the panel and
    draws in as it goes. All of that is lit by its height, a fake ambient
    occlusion - what is tucked down is darker, what stands up is brighter -
    so a pressed pedal goes into shadow, and a rolled gate darkens along its
    low side. The dials are left out of it: a car's gauges are lit from
    within.

    Brightness is what separates a guide from a reading. The gate and the
    arcs sit at a tenth or so of full, and only the needles and the knob are
    fully lit.

    The upshift warning breathes rather than blinks: the redline zone
    brightens and falls back a few times a second while the needle stays fully
    lit above it, so the alarm is unmistakable and the reading is never
    interrupted.

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

    # Section heights, top to bottom. The pedals take whatever is left above
    # FLOOR, so only these plus the gaps are fixed.
    DIAL_ROWS = 7
    H_ROWS = 9
    MIN_PEDAL_ROWS = 2
    # None between the two dials: tilted back, neither reaches the whole of
    # its rows, and the arc of one sits under the ends of the other.
    GAP = 0
    DIAL_GAP = 2        # under them, before the gearbox
    SHIFT_GAP = 3       # under that, before the pedals at the foot of the panel
    # Dark rows left under the pedals. A pedal filling to the panel's edge has
    # its foot cut off by it, and pressed, the foot is what shows it going.
    FLOOR = 1

    # The dials. Both are the same circle, as wide as the panel, swept
    # DIAL_SWEEP either side of straight up - most of a turn, the way a real
    # dial is - and tipped DIAL_TILT back out of the panel about the pivot, so
    # the top of the arc stands up off it and the bottom sinks in. The arm is
    # drawn from ARM_FROM of the way out to the arc.
    DIAL_SWEEP = math.radians(125.0)
    DIAL_TILT = math.radians(50.0)
    ARM_FROM = 0.3

    KNOB_HEIGHT = 3.0   # px the gear knob stands above its gate
    # A pedal hinges at its top, as the real ones do: pressed all the way its
    # foot sinks PEDAL_DEPTH into the panel and its top a share of that.
    PEDAL_DEPTH = 3.0
    PEDAL_HINGE = 0.35

    PEDALS = ('clutch', 'brake', 'throttle')   # left to right, as in the footwell

    # Brightness. Guides an order below readings; see the class docstring.
    TRACK = 44          # the arc an arm sweeps
    REDLINE = 50        # added to it past the upshift point
    WARN_WASH = 45      # and to the whole arc, breathing, while it warns
    GATE = 26           # the H pattern the knob runs over, flat on the panel
                        # (it, the lever and the knob are shaded by height)
    NODE = 72           # where a gear sits on it
    LEVER = 40          # the stalk from the gate up to the knob
    NEEDLE = 255
    BG_CEILING = 120    # the most any guide may reach, so the needle always tells
    MARKER = 255
    TRAIL = 0.5         # share of full the knob's wake carries
    TRAIL_TAU = 0.11    # seconds it takes to fade by e
    PEDAL = 170         # a pedal all the way up, lit; pressing it only darkens it

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
    # Seconds over which a vertical acceleration that does not go away is
    # taken for the level rather than for a bump. A car cannot keep rising or
    # falling faster and faster, so nothing real is lost, and a game that
    # left gravity in after all would only ever shift the level.
    HEAVE_TAU = 1.0
    PACKET_MIN = 0.002     # bounds on the gap between two packets that a
    PACKET_MAX = 0.5       # speed difference may be divided by
    IMPACT_REARM = 0.7     # share of impact_g it has to drop under to hit again
    IMPACT_GAP = 0.25      # seconds between two hits, at the least
    # Shorter than this share of a hit across the panel and the hit is taken
    # as coming from under or over it - a landing - and splashes the middle.
    IMPACT_FLAT = 0.35
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

    def __init__(self, height=34, max_speed=90.0, speed_knee=12.0,
                 redline_floor=4500.0,
                 shift_fraction=0.92, shift_release=0.05, shift_speed=30.0,
                 neutral_return=0.8, flash_hz=8.0, motion=True, lean_px=1.5,
                 lean_g=1.1, heave_px=1.5, impact_g=4.0, shake_px=2.5,
                 tilt_gain=2.5, level_tau=3.0, max_tilt_deg=20.0,
                 yaw_gain=1.0, yaw_tau=0.12, max_yaw_deg=10.0,
                 roll_sign=1.0, pitch_sign=1.0, yaw_sign=1.0,
                 motionsim_ip="127.0.0.1", motionsim_port=4444):
        """
        :param max_speed: m/s at the far end of the speedometer. Only a
            starting point - the scale grows if it is ever exceeded, so the
            needle never sits pinned and silent.
        :param speed_knee: m/s where the speedometer's scale turns from
            linear to logarithmic. The dial is read as log(1 + speed / knee),
            so every speed well under the knee takes an even share of the
            dial, and past it each doubling does: a van at 20 m/s and a
            supercar at 80 are both well round the dial, and the difference
            between 5 and 10 is as easy to see as between 40 and 80. Smaller
            spreads the bottom end wider.
        :param redline_floor: rpm the tachometer assumes as full scale until it
            has watched the engine rev higher. Keeps a car that has only idled
            from scaling 900rpm across the whole dial.
        :param shift_fraction: share of the learned redline that starts the
            tachometer breathing, and where its marked zone begins.
        :param shift_release: extra share it has to drop back through before the
            warning stops, so a needle sitting on the threshold does not stutter
            in and out of it.
        :param shift_speed: pixels/sec the knob travels along the H pattern.
        :param neutral_return: seconds the lever has to sit in neutral before it
            counts as parked there rather than passing through, and slides back
            to the centre of the gate plane. Longer than any shift, shorter than
            any deliberate stop in neutral.
        :param flash_hz: breaths per second of the upshift warning. Capped at
            whatever the measured frame rate can actually resolve, so the same
            setting reads as a pulse rather than as an aliased stutter on stock
            firmware, which draws at a tenth of the patched firmware's rate.
        :param motion: whether the car's motion moves the dash at all.
        :param lean_px: pixels the dash slides across or along the panel at
            lean_g - away from the way the car is accelerating, as whatever is
            loose in it would.
        :param lean_g: acceleration, in g, that slides it that far.
        :param heave_px: pixels it rises out of the panel or sinks into it at
            lean_g, up or down, again the opposite way to the car.
        :param impact_g: acceleration no tyre can produce, so anything past it
            is taken for a collision: it kicks the dash away from the blow and
            sends a ripple across the panel from the edge the blow came from.
        :param shake_px: the furthest the dash is ever allowed to slide, so a
            heavy crash cannot push a gauge off the panel.
        :param tilt_gain: how many times over the dash rolls and pitches with
            the car. A car leans a few degrees in a corner, which on a panel
            nine pixels wide would be nothing at all.
        :param level_tau: seconds over which a roll or pitch the car holds is
            let go of, so a long hill or a banked road does not leave the dash
            tipped for as long as it lasts, while a corner's lean is kept.
        :param max_tilt_deg: the furthest it ever rolls or pitches.
        :param yaw_gain: how many times over it turns with the car.
        :param yaw_tau: seconds a turn is held for before it is let go of: a
            steady corner holds the dash turned by the car's rate of turn
            times this, and a flick turns it and hands it straight back.
        :param max_yaw_deg: the furthest it ever turns.
        :param roll_sign, pitch_sign, yaw_sign: -1 to turn any of them the
            other way, should a game version ever count one differently.
        :param motionsim_ip: address MotionSim is sent to.
        :param motionsim_port: and its port, 4444 as BeamNG's own settings have
            it. MotionSim is a switch of its own in the game (Options > Other >
            Protocols, beside OutGauge); with it off, everything else here
            carries on, the dash only sliding along the car on OutGauge's
            speed, and never tipping.
        """
        super().__init__(height)
        # Both are divisors every frame, so a zero from config.json would take
        # the panel out rather than just mis-scale it.
        self.max_speed = max(max_speed, 1.0)
        self.speed_knee = max(speed_knee, 0.1)
        self.redline_floor = max(redline_floor, 1.0)
        self.shift_fraction = shift_fraction
        self.shift_release = shift_release
        self.shift_speed = shift_speed
        self.neutral_return = neutral_return
        self.flash_hz = flash_hz
        self.motion = motion
        self.lean_px = lean_px
        self.lean_g = max(lean_g, 0.1)
        self.heave_px = heave_px
        self.impact_g = impact_g
        self.shake_px = shake_px
        self.tilt_gain = tilt_gain
        self.level_tau = max(level_tau, 0.05)
        self.max_tilt = math.radians(max_tilt_deg)
        self.yaw_gain = yaw_gain
        self.yaw_tau = max(yaw_tau, 0.01)
        self.max_yaw = math.radians(max_yaw_deg)
        self.signs = (roll_sign, pitch_sign, yaw_sign)
        if motion:
            motionsim_reader.listen_on(motionsim_ip, motionsim_port)

        self.tach_y = 0
        self.speed_y = self.tach_y + self.DIAL_ROWS + self.GAP
        self.h_y = self.speed_y + self.DIAL_ROWS + self.DIAL_GAP
        self.pedal_y = self.h_y + self.H_ROWS + self.SHIFT_GAP
        self.pedal_rows = height - self.FLOOR - self.pedal_y
        if self.pedal_rows < self.MIN_PEDAL_ROWS:
            raise ValueError(
                f"BeamngDashModule needs at least "
                f"{self.pedal_y + self.MIN_PEDAL_ROWS + self.FLOOR} rows, "
                f"got {height}")

        # H pattern rows: the gear ends sit one row in from each edge of the
        # section so the 3x3 knob still fits when parked on them.
        self.top_y = self.h_y + 1
        self.bot_y = self.h_y + self.H_ROWS - 2
        self.rail_y = (self.top_y + self.bot_y) // 2

        # Column geometry depends on the width, which only arrives at render.
        self.gates = None
        self.nodes = {}
        self._wake = None        # the knob's fading trail, panel-sized
        self._ripple = None      # what a collision sends across the panel

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

        self._shove = [0.0, 0.0, 0.0]     # px the whole dash is moved by
        self._shove_v = [0.0, 0.0, 0.0]
        self._accel = [0.0, 0.0, 0.0]     # m/s^2 in the panel's terms, smoothed
        self._heave_level = 0.0           # vertical acceleration taken as level
        self._tilt = [0.0, 0.0, 0.0]      # roll, pitch, yaw the dash has taken, rad
        self._angles = None               # the car's, last frame
        self._armed = True                # whether the next hit counts
        self._last_hit_at = None
        self._impacts = 0
        self._motionsim = False           # whether MotionSim is sending, for the log
        self._motion_at = None
        self._last_packet_at = None
        self._last_speed = 0.0

        self._live = False
        self._last_live_at = None
        self._seen_gears = set()   # for the log trail, see _note()

    # ── geometry ─────────────────────────────────────────────────────────────
    def _ensure_geometry(self, width):
        """Place the three gates and the gear each one holds, and the dials.

        At the panel's 9 columns the gates land on 1, 4 and 7, which is exactly
        three 3-wide knob slots side by side with nothing left over.
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
        self._ripple = Ripple(width, self.height, width / 2.0, self.height / 2.0)

        # The dial is as wide as the panel, and its pivot sits low enough in
        # its rows that the top of the arc, raised and so seen a little out
        # from the middle, lands squarely on the section's first row.
        rest = _Scene(self.height, width)
        self.dial_r = (width - 1) / 2.0
        self.dial_x = rest.cx
        rise = self.dial_r * math.sin(self.DIAL_TILT)
        self.dial_pivot = {
            y0: rest.at_rest(self.dial_x, y0, rise)[1]
            + self.dial_r * math.cos(self.DIAL_TILT)
            for y0 in (self.tach_y, self.speed_y)}
        self._rest = rest

    def _node_for(self, gear):
        """Where an engaged forward gear parks the knob. Reverse and neutral
        are decided in _update_marker and never reach this.

        Seventh and eighth in the odd car that has them share sixth's slot: a
        6-speed pattern has nowhere else to put them, and the alternative is a
        knob that vanishes off the top of the box.
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
        outright, instead of leaving a knob that mysteriously never moves.
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
        """Move and tip the whole dash with the car.

        The panel is the car seen from above: its top is the front, its right
        the right, and out of it is up. Two things are taken from the car's
        motion, and kept apart.

        Acceleration slides the dash, the opposite way to the car, the way
        whatever is loose in it goes: under power it slides down the panel,
        to the rear; braking throws it up the panel; a corner throws it to
        the outside; a crest lifts it out of the panel and a compression
        presses it in. It hangs on a spring throughout, so a slide arrives
        with a little overshoot and a hit rings out rather than snapping back.

        Rotation tips it the same way the car goes: roll right and its right
        side goes down into the panel, pitch the nose up and its top comes up
        out of it, turn right and it turns clockwise. The dash is flat, so on
        its own a tip would barely show; it is the dials standing up off the
        panel and the knob raised on its lever that swing across it.

        Both come out of MotionSim. Without it, the speed OutGauge sends
        differentiated between packet arrivals still says how hard the car is
        speeding up or slowing down, and the dash slides along the car on
        that alone and never tips.

        Anything past impact_g is not something tyres can do to a car, so it
        is taken for a collision rather than for driving: it kicks the dash
        away from the blow, harder than a slide can move it, and sends a
        ripple across the panel from the edge the blow came from.
        """
        if not self.motion:
            self._shove = [0.0, 0.0, 0.0]
            self._tilt = [0.0, 0.0, 0.0]
            return

        now = time.monotonic()
        motion = get_motion()
        got = motion.get('received_at')
        sim = got is not None and now - got < self.STALE_AFTER
        if sim != self._motionsim:
            self._motionsim = sim
            logging.info("BeamngDash: MotionSim %s",
                         "is up, sliding and tipping on it" if sim
                         else "has gone quiet")
            self._angles = None
            self._motion_at = None

        felt = None     # the acceleration of this frame, if there is a new one
        if sim:
            if got != self._motion_at:
                self._motion_at = got
                right, forward, down = motion.get('accel', (0.0, 0.0, 0.0))
                raw = [right, -forward, -down]      # the panel's x, y and z
                self._heave_level += ((raw[2] - self._heave_level)
                                      * (1.0 - math.exp(-dt / self.HEAVE_TAU)))
                raw[2] -= self._heave_level
                felt = raw
            self._tip(dt, motion)
        else:
            self._tip(dt, None)
            felt = self._accel_from_speed(tel, live)

        if felt is not None:
            size = math.sqrt(sum(a * a for a in felt))
            limit = self.impact_g * self.G
            if size > limit and self._armed and (
                    self._last_hit_at is None or now - self._last_hit_at > self.IMPACT_GAP):
                self._hit(felt, size / limit, now)
            elif size < limit * self.IMPACT_REARM:
                self._armed = True
            if size > limit:
                felt = [0.0, 0.0, 0.0]  # a collision is not a lean
            weight = 1.0 - math.exp(-max(dt, self.PACKET_MIN) / self.LEAN_TAU)
            for axis in range(3):
                self._accel[axis] += (felt[axis] - self._accel[axis]) * weight
        elif not live and not sim:
            for axis in range(3):
                self._accel[axis] *= math.exp(-dt / self.LEAN_TAU) if dt else 1.0

        full = self.lean_g * self.G
        reach = (self.lean_px, self.lean_px, self.heave_px)
        target = [-max(-1.0, min(1.0, self._accel[axis] / full)) * reach[axis]
                  for axis in range(3)]
        for step in self._steps(dt):
            for axis in range(3):
                pull = -self.SPRING * (self._shove[axis] - target[axis])
                self._shove_v[axis] += (pull - self.DAMPING * self._shove_v[axis]) * step
                self._shove[axis] += self._shove_v[axis] * step
                if abs(self._shove[axis]) > self.shake_px:
                    self._shove[axis] = math.copysign(self.shake_px, self._shove[axis])
                    self._shove_v[axis] = 0.0

    def _accel_from_speed(self, tel, live):
        """How hard the car is speeding up, from OutGauge's speed, as the
        panel's x, y and z - or None if no new packet has come since last
        frame. OutGauge carries no accelerometer, so this is all it can say."""
        received = tel.get('received_at') if live else None
        if received is None or received == self._last_packet_at:
            return None
        gap = None if self._last_packet_at is None else received - self._last_packet_at
        speed = tel.get('speed', 0.0)
        felt = None
        if gap is not None and self.PACKET_MIN <= gap <= self.PACKET_MAX:
            felt = [0.0, -(speed - self._last_speed) / gap, 0.0]
        self._last_packet_at = received
        self._last_speed = speed
        return felt

    def _tip(self, dt, motion):
        """Follow the car's roll, pitch and yaw, letting go of what it holds.

        What turns the dash is the car turning, not where it points: each
        frame's change in each angle is added on, and what has been added
        leaks away - slowly for roll and pitch, so a corner's lean stays and
        only a long hill is forgotten, and quickly for yaw, which in a corner
        never stops changing. Taken a frame at a time, a car rolling right
        over onto its roof never jumps from one side of the half-turn to the
        other.
        """
        taus = (self.level_tau, self.level_tau, self.yaw_tau)
        for axis in range(3):
            if dt:
                self._tilt[axis] *= math.exp(-dt / taus[axis])
        if motion is None:
            return
        angles = (motion.get('roll', 0.0), motion.get('pitch', 0.0), motion.get('yaw', 0.0))
        if self._angles is not None:
            for axis in range(3):
                turn = angles[axis] - self._angles[axis]
                turn = (turn + math.pi) % (2.0 * math.pi) - math.pi
                self._tilt[axis] += turn * self.signs[axis]
        self._angles = angles

    def _pose(self):
        """Roll, pitch and yaw the dash is drawn at, radians."""
        roll, pitch, yaw = self._tilt
        clamp = lambda a, limit: max(-limit, min(limit, a))
        return (clamp(roll * self.tilt_gain, self.max_tilt),
                clamp(pitch * self.tilt_gain, self.max_tilt),
                clamp(yaw * self.yaw_gain, self.max_yaw))

    def _hit(self, felt, hit, now):
        """A collision: kick the dash away from the blow, and send a ripple
        across the panel from where the blow came in.

        The car is accelerated away from whatever hit it, so the blow came
        from the other way. Looked for from the middle of the dash, that way
        across the panel meets its edge somewhere, and that is where the
        ripple starts - the top edge for a car driven into a wall, the left
        for one struck on its left side, a corner for anything between. A
        blow mostly from under or over, a landing, has no edge to come from,
        and splashes the middle instead.
        """
        self._impacts += 1
        self._armed = False
        self._last_hit_at = now
        hit = min(hit, 4.0)
        size = math.sqrt(sum(a * a for a in felt))
        for axis in range(3):
            self._shove_v[axis] -= 9.0 * hit * felt[axis] / size

        width, height = self._wake.shape[1], self._wake.shape[0]
        cx = width / 2.0 + self._shove[0]
        cy = height / 2.0 + self._shove[1]
        fx, fy = -felt[0], -felt[1]
        flat = math.hypot(fx, fy)
        if flat < self.IMPACT_FLAT * size:
            at = (width / 2.0, height / 2.0)
        else:
            reach = min(((width if fx > 0 else 0.0) - cx) / fx if abs(fx) > 1e-9 else math.inf,
                        ((height if fy > 0 else 0.0) - cy) / fy if abs(fy) > 1e-9 else math.inf)
            at = (min(max(cx + fx * reach, 0.0), width),
                  min(max(cy + fy * reach, 0.0), height))
        self._ripple.strike(at)
        logging.info("BeamngDash: %.0f m/s^2 reads as a hit, rippling from "
                     "(%.1f, %.1f)", size, at[0], at[1])

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
    def _speed_reading(self, speed):
        """Where `speed` sits on the speedometer, 0 to 1, on its log scale."""
        knee = self.speed_knee
        return (math.log1p(max(speed, 0.0) / knee)
                / math.log1p(self._speed_scale / knee))

    def _on_dial(self, y0, u, r=1.0):
        """The point `r` of the way out along the arm at reading `u`, in the
        scene: a circle swept DIAL_SWEEP either side of straight up, tipped
        back about its pivot so its top stands up off the panel."""
        angle = (2.0 * u - 1.0) * self.DIAL_SWEEP
        across = self.dial_r * r * math.sin(angle)
        up = self.dial_r * r * math.cos(angle)
        return (self.dial_x + across,
                self.dial_pivot[y0] - up * math.cos(self.DIAL_TILT),
                up * math.sin(self.DIAL_TILT))

    def _draw_dial_track(self, scene, y0, zone=None, wash=0.0):
        """A dial's arc, dim, the part past the upshift point brighter."""
        n = 48
        us = np.linspace(0.0, 1.0, n + 1)
        pts = [self._on_dial(y0, u) for u in us]
        mids = (us[1:] + us[:-1]) / 2.0
        levels = np.full(n, self.TRACK + wash)
        if zone is not None:
            levels[mids >= zone] += self.REDLINE
        scene.curve(pts, levels)

    def _draw_needle(self, scene, y0, value):
        """The needle's outer arm, from ARM_FROM of the way out to the arc,
        over the track: long enough to read as an arm pointing, and brighter
        and broader than the track it sweeps."""
        value = min(max(value, 0.0), 1.0)
        scene.line(self._on_dial(y0, value, self.ARM_FROM),
                   self._on_dial(y0, value, 1.0), self.NEEDLE, over=True)

    def _knob(self):
        """Where the knob is over the gate, overshoot and all."""
        return (self._marker[0] + self._settle_dir[0] * self._settle,
                self._marker[1] + self._settle_dir[1] * self._settle)

    def _draw_gate(self, scene):
        """The H pattern, flat on the panel, each gear's end a little
        brighter so the six can be counted with the knob elsewhere."""
        left, _, right = self.gates
        for gx in self.gates:
            scene.line((gx, self.top_y - 0.5, 0.0), (gx, self.bot_y + 0.5, 0.0),
                       self.GATE, shade=True)
        scene.line((left - 0.5, self.rail_y, 0.0), (right + 0.5, self.rail_y, 0.0),
                   self.GATE, shade=True)
        for node in self.nodes.values():
            scene.point((node[0], node[1], 0.0), self.NODE - self.GATE, shade=True)

    def _draw_shifter(self, scene, dt):
        """The knob, standing KNOB_HEIGHT above the gate on its lever.

        Seen from straight on the lever is hidden under the knob; tipped, it
        shows as a short stalk down to the gate, and the knob slides across
        the gate against the rest of the dash.

        The knob leaves a wake, kept in a buffer that fades by e every
        TRAIL_TAU. At 60fps a shift reads as a stroke drawn by something moving,
        which is most of what tells a 1-2 from a 3-4 at this size.
        """
        mx, my, _ = self._rest.at_rest(*self._knob(), self.KNOB_HEIGHT)
        scene.line((mx, my, 0.0), (mx, my, self.KNOB_HEIGHT), self.LEVER, shade=True)

        knob = scene.fresh()
        self._draw_knob(knob, mx, my)
        knob.lay_over()
        if dt > 0.0:
            self._wake *= math.exp(-dt / self.TRAIL_TAU)
        np.maximum(self._wake, knob.buf, out=self._wake)
        scene.buf += self._wake * self.TRAIL
        self._draw_knob(scene, mx, my)

    def _draw_knob(self, scene, mx, my):
        """3x3 about (mx, my) at the knob's height, sized to cover exactly
        nine pixels at rest."""
        side = 3.0 * (scene.EYE - self.KNOB_HEIGHT) / scene.EYE
        scene.quad((mx - side / 2.0, my - side / 2.0, self.KNOB_HEIGHT),
                   (side, 0.0, 0.0), (0.0, side, 0.0), self.MARKER, over=True,
                   shade=True)

    def _draw_reverse(self, scene, width):
        """Reverse gets a letter, not a knob position.

        Three gates fill the width, and every one of their six ends is a forward
        gear, so there is no free slot to park reverse in. An R stands where the
        knob would, at its height, and can't be mistaken for a gear.
        """
        glyph = tiny_font['R']
        scale = max(1, min(self.H_ROWS // len(glyph), width // len(glyph[0])))
        x0 = (width - len(glyph[0]) * scale) / 2.0 - 0.5
        y0 = self.h_y + (self.H_ROWS - len(glyph) * scale) / 2.0 - 0.5
        side = scale * (scene.EYE - self.KNOB_HEIGHT) / scene.EYE
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == '1':
                    corner = self._rest.at_rest(x0 + col * scale, y0 + row * scale,
                                                self.KNOB_HEIGHT)
                    scene.quad(corner, (side, 0.0, 0.0), (0.0, side, 0.0),
                               self.MARKER, over=True, shade=True)

    def _pedal_layout(self, width):
        """Where the three lights sit, left to right, with a dark column
        between them. A spare column goes to the middle one, which is the
        brake - the pedal a glance is most often after.
        """
        n = len(self.PEDALS)
        base, extra = divmod(width - (n - 1), n)
        widths = [base] * n
        outward = sorted(range(n), key=lambda i: abs(i - (n - 1) / 2.0))
        for i in range(extra):
            widths[outward[i % n]] += 1
        spans, x = [], 0
        for w in widths:
            spans.append((x, x + w - 1))
            x += w + 1
        return spans

    def _draw_pedals(self, scene, width, tel):
        """Clutch, brake and throttle as three lit faces that are pressed
        down into the panel, and into its shadow, with the pedal.

        Up, a pedal sits level with the panel and fully lit. Pressed, it
        hinges down about its top edge, the way the pedal itself swings, so
        its foot goes deepest; lit by its height, it darkens as it goes, and
        its foot darkest of all. Nothing has to be added to say a pedal is
        pressed: a panel with three dark wells at the bottom of it is a driver
        on all three.

        The perspective does the rest: a sunken face is seen a little smaller
        and drawn in toward the middle of the panel.
        """
        rows = self.pedal_rows
        for (x0, x1), pedal in zip(self._pedal_layout(width), self.PEDALS):
            value = min(max(tel.get(pedal, 0.0), 0.0), 1.0)
            sink = self.PEDAL_DEPTH * value
            top = -sink * self.PEDAL_HINGE
            scene.quad((x0 - 0.5, self.pedal_y - 0.5, top),
                       (x1 - x0 + 1.0, 0.0, 0.0), (0.0, rows, -sink - top),
                       self.PEDAL, over=True, shade=True)

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

        scene = _Scene(self.height, width)
        roll, pitch, yaw = self._pose()
        scene.pose(roll, pitch, yaw, self._shove)

        warning = self._shift_up(tel, gear, live, now)
        pulse = self._warn_pulse(now) if warning else 0.0
        # Guides first, all of them held under BG_CEILING, so the needles and
        # the knob stay the brightest things on the panel at every point in
        # the warning's breath. The rest of the tachometer's arc comes up with
        # its zone, so the warning is a whole dial breathing rather than a
        # bright edge appearing.
        self._draw_dial_track(scene, self.tach_y, zone=self.shift_fraction,
                              wash=pulse * self.WARN_WASH)
        self._draw_dial_track(scene, self.speed_y)
        self._draw_gate(scene)
        np.clip(scene.buf, 0.0, self.BG_CEILING, out=scene.buf)

        self._draw_pedals(scene, width, tel)
        self._draw_needle(scene, self.tach_y, tel.get('rpm', 0.0) / self._redline)
        self._draw_needle(scene, self.speed_y, self._speed_reading(tel.get('speed', 0.0)))
        scene.lay_over()
        if gear <= 0:
            self._draw_reverse(scene, width)
        else:
            self._draw_shifter(scene, dt)
        scene.lay_over()

        pixels = scene.image()
        if self._ripple.active:
            self._ripple.advance(dt)
            pixels = self._ripple.apply(pixels)
        return Image.fromarray(pixels, 'L')
