import math
import time

import numpy as np

from utils.sound_medium import SoundMedium


class Ripple:
    """The panel as a pool, and something just dropped into it at the jack.

    The same kind of surface as the sound visualizer's: a damped wave equation
    with reflecting edges, simulated at SCALE times the panel's resolution. A
    plug going in drives it hard at the jack for a couple of quick cycles, and
    rings of crests and troughs race out across the panel - under a second to
    the far corner - bounce off the edges and interfere as they fade.

    Whatever the panel is showing is the floor of the pool, and moves with the
    water over it. A crest pushes the picture under it out, away from the
    jack, and the trough behind it pulls it back in, so each ring passing
    shoves everything outward and back, the way a floating leaf bobs out and
    in as a wave goes by. Nothing is lit or dimmed: the pixels already on the
    panel are only moved about.

    Only the picture moves: nothing in it is carried anywhere, so once the
    water is still the panel is exactly what it was.
    """

    SCALE = 3                   # simulation cells per panel pixel, each way
    # Seconds per simulation step. The surface is stepped explicitly, and past
    # about 0.7 cells a step it blows up: at SPEED that needs a step this short,
    # and it holds up to 160 cells/s, which is as far as pixeled-tune goes.
    STEP = 1 / 240

    # The surface. Speeds are in cells per second.
    SPEED = 120.0               # 40 px/s: the far corner is ~0.6 s away
    DAMPING = 5.0               # 1/s; how quickly the rings die away
    # 1/s²; pulls the surface back to level. Edges that reflect let no water
    # out, so without it whatever the drive left over stood as a level raised
    # across the whole panel, and held everything pushed out for good.
    #
    # The slowest thing the surface can do is bob as a whole, at the square
    # root of this, and that bob is what is left over once the rings have
    # gone. Set to DAMPING squared over four it is critically damped: the
    # fastest it can come back to level without bouncing, and the picture is
    # done moving in about 1.8 s rather than 2.3. Softer and the bob is
    # overdamped, which sounds calm but is slower - it cannot swing back, so
    # it only leaks away, and leaves a long tail of everything sitting
    # slightly out of place.
    SPRING = 6.25
    FREQ = 3.4                  # Hz the plug drives the water at
    CYCLES = 1.0                # of driving
    ATTACK = 0.04               # seconds the drive takes to reach full
    DECAY = 0.2                 # seconds it takes to die away by 1/e
    SOURCE = 5.0                # cells; the size of the splash
    DRIVE = 5359.0              # push, per unit of envelope
    QUIET = 0.05                # height and speed under which the water is still
    # Seconds a frame may spend on the water, for the reason SoundMedium.BUDGET
    # gives: a slow frame has more to catch up on, which makes it slower again.
    # Past this the ripple runs slow rather than holding the panel up.
    BUDGET = 0.004

    # The picture on the floor, pushed out by crests and pulled in by troughs.
    PUSH = 10.0                 # cells moved per unit of height
    # The most anything moves, in cells. A tanh approaches it smoothly, which
    # also keeps a steep wave from folding the picture over itself.
    PUSH_MAX = 15.0
    # Moving the picture by a fraction of a pixel splits a pixel's light
    # between two of them, which fattens and dims a one-pixel letter stroke,
    # and a whole panel of letters left half a pixel over reads as a blur.
    #
    # So the picture is pulled toward the pixels it is made of: of the
    # fraction of a pixel it is moved by, the part near a whole pixel is
    # rounded to it and only what is left near halfway stays between two. At
    # GRID_PULL 0 nothing is rounded, and the picture slides through the greys
    # as the water moves it; at 1 everything is, and it moves a whole pixel at
    # a time in black and white; in between, what is nearly in place clicks
    # into place and what is genuinely between pixels is still drawn between
    # them. Nothing is thresholded on how big the wave is: a ripple dying away
    # moves the picture less and less, so more and more of it is already
    # nearly in place, and the panel comes back to itself pixel by pixel
    # instead of all at once when the water finally goes still.
    GRID_PULL = 0.7             # 0 all greys, 1 whole pixels only

    def __init__(self, width, height, jack_x, jack_y):
        """`jack_x` and `jack_y` are where the plug goes in, in pixels, as
        fractional coordinates of the panel: (width, y) is on its right edge."""
        s = self.SCALE
        self.width, self.height = width, height
        self.w, self.h = width * s, height * s
        self._ys, self._xs = np.mgrid[0:self.h, 0:self.w].astype(np.float64)
        self._py, self._px = np.mgrid[0:height, 0:width].astype(np.float64)
        # Cell centres sit at half cells, so a pixel coordinate p is cell p*s - 0.5.
        self._jack = (jack_x * s - 0.5, jack_y * s - 0.5)
        self._shape()
        self.stop()

    def _shape(self):
        """The splash, and which way is out from it: SOURCE across. Worked out
        again on every strike, since pixeled-tune can change SOURCE."""
        dx, dy = self._xs - self._jack[0], self._ys - self._jack[1]
        r = np.sqrt(dx * dx + dy * dy)
        self._splash = np.exp(-(r * r) / (2 * self.SOURCE ** 2))
        # Which way is out, faded to nothing right at the jack, where out is
        # every direction at once. Per pixel, since it is the picture this
        # moves, not the water.
        s = self.SCALE
        dx = self._px - (self._jack[0] + 0.5) / s + 0.5
        dy = self._py - (self._jack[1] + 0.5) / s + 0.5
        r = np.sqrt(dx * dx + dy * dy)
        fade = np.minimum(r * s / self.SOURCE, 1.0) / np.maximum(r, 1e-9)
        self._out_x, self._out_y = dx * fade, dy * fade

    def stop(self):
        self.height_field = np.zeros((self.h, self.w))
        self.velocity = np.zeros((self.h, self.w))
        self._driven = None         # seconds since the plug went in, while driving
        self._carry = 0.0
        self.active = False

    def strike(self):
        """The plug going in. Adds to whatever is still rippling."""
        self._shape()
        self._driven = 0.0
        self.active = True

    def advance(self, dt):
        if not self.active:
            return
        self._carry += min(dt, 0.1)
        h, v = self.height_field, self.velocity
        drive_time = self.CYCLES / self.FREQ
        deadline = time.monotonic() + self.BUDGET
        while self._carry >= self.STEP:
            self._carry -= self.STEP
            dt = self.STEP
            accel = (self.SPEED ** 2 * SoundMedium._laplacian(h)
                     - self.DAMPING * v - self.SPRING * h)
            if self._driven is not None:
                # Hard at once, as a plug goes in, and dying away over the
                # cycles, so the first ring is the biggest.
                t = self._driven
                envelope = (1 - math.exp(-t / self.ATTACK)) * math.exp(-t / self.DECAY)
                accel += (self.DRIVE * envelope * math.sin(2 * math.pi * self.FREQ * t)
                          * self._splash)
                self._driven = t + dt if t + dt < drive_time else None
            v += dt * accel
            h += dt * v
            if time.monotonic() >= deadline:
                self._carry = 0.0       # given up, not owed
                break
        if (self._driven is None and np.abs(h).max() < self.QUIET
                and np.abs(v).max() < self.QUIET * self.SPEED):
            self.stop()

    def apply(self, pixels):
        """`pixels`, height x width of 0-255, as seen through the water now."""
        if not self.active:
            return pixels
        s = self.SCALE
        floor = pixels.astype(np.float64)
        # The water as the pixels see it: what the surface does across a pixel
        # is what moves it, so the picture is moved on its own grid rather than
        # on the simulation's. Moved on the simulation's, a pixel's worth of
        # cells could round to different pixels, and a fully pulled picture
        # still came out with greys in it.
        h = self.height_field.reshape(self.height, s, self.width, s).mean(axis=(1, 3))
        push = (self.PUSH_MAX * np.tanh(self.PUSH * h / self.PUSH_MAX)) / s

        # Each axis separately, since it is a whole pixel each way that leaves
        # the picture crisp. The fraction of a pixel is stretched about its
        # middle by 1 / (1 - GRID_PULL) and held to the pixel either side, so
        # the pull reaches further into the fraction the stronger it is: at 0
        # the fraction is untouched, and at 1 it is 0 or 1 and nothing lands
        # between pixels at all.
        keep = max(1.0 - self.GRID_PULL, 1e-6)
        def sample_at(along, moved, size):
            # Worked backwards: each pixel shows what the push moved to it, and
            # what comes in from past an edge is the picture reflected in it.
            p = np.abs(along - moved)
            p = np.where(p > size - 1, 2 * (size - 1) - p, p).clip(0, size - 1.001)
            whole = np.floor(p)
            return whole.astype(int), np.clip((p - whole - 0.5) / keep + 0.5, 0.0, 1.0)

        x0, fx = sample_at(self._px, push * self._out_x, self.width)
        y0, fy = sample_at(self._py, push * self._out_y, self.height)
        moved = ((floor[y0, x0] * (1 - fx) + floor[y0, x0 + 1] * fx) * (1 - fy)
                 + (floor[y0 + 1, x0] * (1 - fx) + floor[y0 + 1, x0 + 1] * fx) * fy)
        return np.rint(np.clip(moved, 0, 255)).astype(np.uint8)


class Breach:
    """The panel as a pressurised vessel, holed at the jack.

    Everything on the panel is fluid, and the moment the plug comes out it all
    rushes for the hole. The flow is worked out once (see _solve_flow), with the
    side walls letting nothing through and the hole open to the outside, and
    the fluid runs down its pressure. So everything converges on the jack,
    carried along the panel toward its row and bending round to get there,
    faster the closer it is. The vessel's pressure also drives it harder the
    longer it drains, so nothing is left behind for long.

    On its own that flow is glassy: every streamline runs to the hole and
    nothing in it ever meets anything else. What a real vessel does is shed
    eddies. Fluid cannot slip along a wall - it is held still at it and moves
    freely a little way off - so every wall carries a layer of spin, and as the
    flow accelerates toward the hole those layers peel away and roll up.

    So the eddies are carried explicitly, as vortices: each with a place, a
    spin and a core, shed off the walls as fast as the fluid runs along them,
    drifting with the flow that made them and with each other, and dying away
    as their spin is lost. Which way one turns is decided by the wall it came
    off and the way the fluid was going: the same rule that makes the layer,
    not a coin toss. Every one of them turns the fluid around it, which is how
    the flow reaches itself - a stream passing an eddy is swung around it, two
    of them braid, and what they do next depends on where they have pushed each
    other. They are shed hardest where the fluid runs fastest, so the churn
    builds as the drain pulls harder and eases as the vessel empties.

    What is on the panel when the plug comes out is broken into particles,
    SUB x SUB to a pixel, each carrying its share of that pixel's light. They
    have a little inertia, so they lag the flow and overshoot its bends, and
    they coast through the eddies rather than turning on the spot with them.

    Nothing pushes them apart: the flow they ride cannot be squeezed, so what
    narrows one way stretches the other and the picture keeps its bulk as it
    is funnelled. A force to hold them apart was tried, and with the flow
    already doing it, all it did was slow the drain and puff the streams out.
    Packed together in the throat they brighten, and what goes out lights the
    jack as it leaves. Each pixel fades rather than going dark the moment it
    empties, so the streams leave streaks.
    """

    GRID = 2                    # pressure cells per pixel, each way
    OPENING = 1.0               # px of wall a hole spans, unless told otherwise
    SUB = 3                     # particles per pixel, each way
    STEP = 1 / 240              # seconds per step; fast particles cross pixels quickly
    SPEED = 28.0                # px/s the picture is drawn in at, typically
    MAX_SPEED = 110.0           # px/s
    RAMP = 0.05                 # seconds for the breach to get going
    PRESSURE_RISE = 3.0         # how much harder it drives per second draining
    INERTIA = 0.035             # seconds a particle takes to follow the flow
    # The eddies shed off the walls. SPIN is the spin itself, and what it
    # comes to is how fast an eddy turns what is beside it: fastest a core out,
    # at SPIN / (2 x CORE) px/s. Set so that is a fraction of the flow it is
    # perturbing - much more and an eddy flings whatever passes it across the
    # panel, which reads as scripted rather than as fluid, because nothing that
    # fast happens for any reason the eye can see.
    SHED = 0.09                 # average seconds between one and the next
    SPIN = 100.0                # px²/s of spin per unit of the flow's speed
    CORE = 1.1                  # px; inside this a vortex turns as one piece
    SPIN_FADE = 0.4             # seconds a vortex's spin takes to fall by 1/e
    EDDIES = 12                 # the most kept at once, oldest dropped first
    EXIT = 0.8                  # px from the hole at which a particle is out
    TRAIL = 0.03                # seconds a pixel takes to fade once emptied
    GLOW = 0.08                 # seconds the jack's light takes to fade
    GLOW_GAIN = 0.35            # light at the jack per unit of light gone out
    GIVE_UP = 2.5               # seconds after which anything left is let out
    # Seconds a frame may spend on the fluid, for the reason SoundMedium.BUDGET
    # gives: a frame that is late has more to catch up on, which makes it later.
    BUDGET = 0.006

    def __init__(self, width, height, jack_x, jack_y, opening=None, seed=11):
        """The hole is `opening` pixels of the wall, centred on the port."""
        self.width, self.height = width, height
        self.jack = (float(jack_x), float(jack_y))
        self.opening = float(self.OPENING if opening is None else opening)
        self._rng = np.random.default_rng(seed)
        rng = self._rng
        self._flow = self._solve_flow()
        self._eddies = np.zeros((0, 3))          # x, y, spin
        self._shed_at = 0.0                      # when the next one is due
        self._walls = self._wall_layers()
        self._particles = np.zeros((0, 6))       # x, y, vx, vy, light, inertia
        self._rng = rng
        # How much of each pixel row of the wall the hole takes, for the light
        # of what leaves to show in.
        low, high = self.jack[1] - self.opening / 2, self.jack[1] + self.opening / 2
        edges = np.arange(self.height + 1, dtype=np.float64)
        covered = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0, None)
        self._mouth = covered / max(covered.sum(), 1e-9)
        self.active = False

    def _solve_flow(self):
        """The fluid's velocity at each pressure cell, in px/s, as (vx, vy).

        Fluid that does not compress, drawn out through the hole, with emptiness
        coming in evenly behind it across every wall but the hole's own:
        Laplace's equation for the pressure, zero just outside the hole, with a
        fixed slope at those walls. Small enough to solve directly.

        Let in across the ends alone, the stream coming down met the one coming
        up along the hole's own row, and the two cancelled: everything out
        along that row sat in water moving at a fiftieth of the speed it moved
        at anywhere else, and hung against the far wall until the vessel's
        pressure had risen enough to drag it off. Letting it in across the far
        wall as well leaves the slow places in the corners, where a vessel does
        empty last.

        Not a gas expanding out of the hole, which is what a real breach is:
        that thins everything out where it is, and the picture spread into a
        grey haze over the whole panel before any of it had reached the jack.
        Kept whole, the picture is carried along the panel toward the jack and
        squeezed through it, stretching as it goes.
        """
        g = self.GRID
        gw, gh = self.width * g, self.height * g
        size = gw * gh
        jy = self.jack[1]
        opening = [i for i in range(gh)
                   if abs((i + 0.5) / g - jy) <= self.opening / 2]
        a = np.zeros((size, size))
        rhs = np.zeros(size)
        for i in range(gh):
            for j in range(gw):
                k = i * gw + j
                for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= ni < gh and 0 <= nj < gw:
                        a[k, ni * gw + nj] += 1
                        a[k, k] -= 1
                    elif nj == gw and i in opening:
                        a[k, k] -= 1        # open to the outside, at nothing
                    elif ni < 0 or ni >= gh or nj < 0:
                        rhs[k] -= 1         # emptiness let in across that wall
        p = np.linalg.solve(a, rhs).reshape(gh, gw)
        # Downhill, with the side walls standing in for themselves, the ends
        # sloping as they push, and the outside past the hole at nothing.
        right = np.concatenate((p[:, 1:], p[:, -1:]), axis=1)
        right[opening, -1] = 0.0
        left = np.concatenate((p[:, :1] + 1, p[:, :-1]), axis=1)
        down = np.concatenate((p[1:], p[-1:] + 1), axis=0)
        up = np.concatenate((p[:1] + 1, p[:-1]), axis=0)
        vx, vy = -(right - left) / 2, -(down - up) / 2
        # Kept at a typical one px/s across the panel, and multiplied by SPEED
        # where it is used, so how hard the vessel drains can be changed while
        # it does. Measured as the middle of the whole panel's speeds rather
        # than at a wall: how much is let in where depends on how many walls
        # let it in, and the pace of the drain should not.
        scale = 1.0 / max(float(np.median(np.hypot(vx, vy))), 1e-9)
        return vx * scale, vy * scale

    def _wall_layers(self):
        """Where along the walls the fluid runs, and which way its layer of
        spin turns there: (x, y, spin per unit of speed, how fast it runs).

        A wall holds the fluid still at it, so the fluid a little way off is
        shearing against it. Which way that shear turns depends on the side
        the wall is on and the way the flow is going along it, and how strong
        it is depends on how fast: both come straight out of the flow.
        """
        edge = self.CORE            # where the layer sits, off the wall
        along = np.arange(0.5, max(self.width, self.height))
        places = []
        down = along[along < self.height]
        across = along[along < self.width]
        # (points, which way the wall faces) - the sign each wall gives its
        # layer for flow running one way along it.
        for x, y, tangent, sign in (
                (np.full_like(down, edge), down, 'y', +1.0),                    # left
                (np.full_like(down, self.width - edge), down, 'y', -1.0),       # right
                (across, np.full_like(across, edge), 'x', -1.0),                # top
                (across, np.full_like(across, self.height - edge), 'x', +1.0)):  # bottom
            flow_x, flow_y = self._flow_at(np.asarray(x, dtype=float),
                                           np.asarray(y, dtype=float))
            runs = flow_y if tangent == 'y' else flow_x
            places.append(np.column_stack((x, y, sign * np.sign(runs), np.abs(runs))))
        walls = np.vstack(places)
        # Nothing sheds off the mouth itself, where there is no wall to shear on.
        beside = np.maximum(np.abs(walls[:, 1] - self.jack[1]) - self.opening / 2, 0.0)
        at_mouth = np.hypot(walls[:, 0] - self.jack[0], beside) < self.CORE * 2
        return walls[~at_mouth]

    def _flow_at(self, x, y):
        g = self.GRID
        vx, vy = self._flow
        gx = np.clip(x * g - 0.5, 0, vx.shape[1] - 1.001)
        gy = np.clip(y * g - 0.5, 0, vx.shape[0] - 1.001)
        x0, y0 = gx.astype(int), gy.astype(int)
        fx, fy = gx - x0, gy - y0
        look = lambda f: ((f[y0, x0] * (1 - fx) + f[y0, x0 + 1] * fx) * (1 - fy)
                          + (f[y0 + 1, x0] * (1 - fx) + f[y0 + 1, x0 + 1] * fx) * fy)
        return look(vx) * self.SPEED, look(vy) * self.SPEED

    def _turned_by_eddies(self, x, y, skip=None):
        """What the eddies do to the fluid at each of (x, y): every vortex
        turns what is near it, hardest a core's width out and falling away
        beyond that."""
        e = self._eddies
        if not len(e):
            return np.zeros(len(x)), np.zeros(len(x))
        dx = x[:, None] - e[:, 0]
        dy = y[:, None] - e[:, 1]
        turn = e[:, 2] / (dx * dx + dy * dy + self.CORE ** 2)
        if skip is not None:
            turn[skip] = 0.0        # a vortex does not turn itself
        return (-dy * turn).sum(axis=1), (dx * turn).sum(axis=1)

    def _shed(self, drive):
        """Roll a wall's layer of spin up into a vortex, off whichever stretch
        of wall the fluid is running along fastest."""
        walls = self._walls
        if not len(walls):
            return
        runs = walls[:, 3] * drive
        # Weighted by how fast the fluid runs there, drawn by where the total
        # falls rather than by normalising, which cannot quite sum to one.
        running = np.cumsum(runs)
        if running[-1] <= 0:
            return
        which = int(np.searchsorted(running, self._rng.random() * running[-1]))
        x, y, sign, speed = walls[which]
        speed *= drive
        if speed < 1.0:
            return
        # How strong varies from one to the next, and so does the wait for the
        # next: a wall sheds when it sheds. Timed to a metronome and all of a
        # size, the churn beats like a machine.
        spin = sign * self.SPIN * (speed / self.SPEED) * self._rng.uniform(0.5, 1.5)
        # A little way off the wall, where the layer has left it.
        self._eddies = np.vstack((self._eddies, [x, y, spin]))
        if len(self._eddies) > self.EDDIES:
            self._eddies = self._eddies[-self.EDDIES:]
        wait = self.SHED * self.SPEED / max(runs.max(), 1.0)
        self._shed_at = self._time + wait * self._rng.exponential()

    def _drift_eddies(self, dt, drive):
        """The eddies go where the fluid goes - the drain's flow, and each
        other's turning - and lose their spin as they go."""
        e = self._eddies
        if not len(e):
            return
        flow_x, flow_y = self._flow_at(e[:, 0], e[:, 1])
        turn_x, turn_y = self._turned_by_eddies(e[:, 0], e[:, 1],
                                                skip=np.eye(len(e), dtype=bool))
        e[:, 0] += dt * (flow_x * drive + turn_x)
        e[:, 1] += dt * (flow_y * drive + turn_y)
        e[:, 2] *= math.exp(-dt / self.SPIN_FADE)
        inside = ((e[:, 0] > 0) & (e[:, 0] < self.width)
                  & (e[:, 1] > 0) & (e[:, 1] < self.height)
                  & (np.abs(e[:, 2]) > self.SPIN * 0.02))
        self._eddies = e[inside]

    def start(self, pixels):
        """Hole the vessel with `pixels`, height x width of 0-255, in it.

        A pixel's particles all start at its centre, where a particle draws
        exactly that pixel and no other: spread across it, each lit its
        neighbours a share, and the picture blurred the instant the breach
        began. They part as the flow gets going, each following it a little
        more or less readily than the next.
        """
        n = self.SUB * self.SUB
        rows, cols = np.nonzero(pixels > 0)
        light = np.repeat(pixels[rows, cols].astype(np.float64) / n, n)
        x = np.repeat(cols + 0.5, n)
        y = np.repeat(rows + 0.5, n)
        inertia = self.INERTIA * self._rng.uniform(0.5, 1.5, len(x))
        still = np.zeros_like(x)
        self._particles = np.column_stack((x, y, still, still, light, inertia))
        self._trail = np.zeros((self.height, self.width))
        self._glow = 0.0
        self._gone = 0.0
        self._time = 0.0
        self._carry = 0.0
        self._eddies = np.zeros((0, 3))
        self._shed_at = 0.0
        self.active = True
        self.drained = False

    def advance(self, dt):
        if not self.active:
            return
        self._carry += min(dt, 0.1)
        gone = 0.0
        jx, jy = self.jack
        deadline = time.monotonic() + self.BUDGET
        while self._carry >= self.STEP:
            self._carry -= self.STEP
            dt, t = self.STEP, self._time
            self._time += dt
            p = self._particles
            if not len(p):
                continue
            x, y = p[:, 0], p[:, 1]
            drive = (1 - math.exp(-t / self.RAMP)) * (1 + self.PRESSURE_RISE * t)
            if self._time >= self._shed_at:
                self._shed(drive)
            self._drift_eddies(dt, drive)
            fx, fy = self._flow_at(x, y)
            turn_x, turn_y = self._turned_by_eddies(x, y)
            fx, fy = fx * drive + turn_x, fy * drive + turn_y
            limit = np.maximum(np.hypot(fx, fy) / self.MAX_SPEED, 1.0)
            fx, fy = fx / limit, fy / limit
            follow = 1 - np.exp(-dt / p[:, 5])
            p[:, 2] += (fx - p[:, 2]) * follow
            p[:, 3] += (fy - p[:, 3]) * follow
            p[:, 0] = np.clip(x + p[:, 2] * dt, 0.0, self.width - 1e-6)
            p[:, 1] = np.clip(y + p[:, 3] * dt, 0.0, self.height - 1e-6)
            # The hole is a length of the wall, not a point, so what counts
            # is how far a particle is from the nearest part of it.
            beside = np.maximum(np.abs(p[:, 1] - jy) - self.opening / 2, 0.0)
            out = np.hypot(p[:, 0] - jx, beside) < self.EXIT
            if self._time > self.GIVE_UP:
                out[:] = True
            gone += p[out, 4].sum()
            self._particles = p[~out]
            if time.monotonic() >= deadline:
                self._carry = 0.0       # given up, not owed
                break
        self._gone += gone

    def pixels(self, dt):
        """What the panel shows now, height x width of 0-255."""
        p = self._particles
        field = np.zeros(self.height * self.width)
        if len(p):
            # Spread over the four pixels nearest each, as the eye would blend
            # a particle between them.
            x = np.clip(p[:, 0] - 0.5, 0, self.width - 1.001)
            y = np.clip(p[:, 1] - 0.5, 0, self.height - 1.001)
            x0, y0 = x.astype(int), y.astype(int)
            fx, fy = x - x0, y - y0
            i = y0 * self.width + x0
            for at, share in ((i, (1 - fx) * (1 - fy)), (i + 1, fx * (1 - fy)),
                              (i + self.width, (1 - fx) * fy),
                              (i + self.width + 1, fx * fy)):
                field += np.bincount(at, p[:, 4] * share, field.size)
        field = field.reshape(self.height, self.width)
        self._trail = np.maximum(field, self._trail * math.exp(-dt / self.TRAIL))
        self._glow = (self._glow * math.exp(-dt / self.GLOW)
                      + self.GLOW_GAIN * self._gone)
        self._gone = 0.0
        out = self._trail.copy()
        out[:, -1] += self._glow * self._mouth
        if not len(p) and out.max() < 2:
            self.active = False
            self.drained = True
        return np.rint(np.clip(out, 0, 255)).astype(np.uint8)
