import math
import time

import numpy as np


class SoundMedium:
    """A small physical world the sound pushes on, drawn as light.

    Three things share the space, each a piece of physics rather than an
    effect, and each coupled to the others:

    - A water surface: a damped wave equation with reflecting edges. Pushed,
      it swells; struck, it rings out in ripples that cross the panel, bounce
      back and interfere. Light passing through it is bent, and gathers where
      the surface curves like a lens - the moving caustics on the floor of a
      pool - and catches on the slopes facing it.
    - Glow: light that lingers. It spreads (diffusion), rises (buoyancy),
      cools (decay), and is carried along by the currents of the surface
      under it.
    - Sparks: particles that are thrown out, surf the slopes of the waves,
      drift up, and twinkle as they burn out.
    - Air: a thump shoves what drifts around it - glow, sparks, the lines on
      the floor - outward and snaps it back, the way a leaf is blown aside by
      something rushing past and sucked straight back after it. Only the
      picture moves: nothing in the world is actually carried anywhere, so it
      all lands back where it was. A punchy hit snaps back quickly and a
      swelling one eases; a boom that rings on holds the push out while it
      does.

    The sound is not matched against cases. Each band of the spectrum acts on
    the world the same way, as a source somewhere in it, and only how it acts
    varies, smoothly, with how high the band is:

    - how wide it pushes: a low band's footprint is wider than the panel, so
      the whole surface heaves and the glow it leaves fills everything, while
      a high band's is a fraction of a pixel;
    - how fast its push oscillates: slow swells for low bands, fast shivers
      for high ones;
    - what it leaves: low bands leave warm glow, high bands throw sparks;
    - where it sits: low sounds settle low, high sounds float, and each band
      sits across the panel where the stereo image puts it, wandering further
      the louder it is.

    Onsets - a band getting louder faster than its recent average - strike the
    surface where the band is, and the sharpest flash there - a burst of light that spreads
    and is gone in a blink, where glow lingers. A drum hit, loud in
    every band at once, strikes everywhere and the whole panel flashes and
    rings; a held chord keeps pushing and waves roll out from it; a hi-hat is
    a spray of glints; a swelling pad lifts the light slowly.

    It is simulated at SCALE times the panel's resolution and averaged down,
    so anything moving glides through the greys between pixels rather than
    stepping from one to the next. Silence adds nothing, and everything
    decays, so it fades to black.
    """

    SCALE = 3                   # simulation cells per panel pixel, each way
    STEP = 1 / 120              # seconds per simulation step
    # Seconds a frame may spend running the world on. Stepping costs what it
    # costs, and how much of the world a frame has to catch up on is however
    # long the last frame took - so when the machine is slow, a frame that is
    # late has more to do, which makes it later still. The panel showed that
    # as a stall of a second or more: measured, one advance() ran for 0.9 s,
    # and the whole panel went to about 3 fps until it caught up. This laptop
    # idles its cores at 544 MHz, where the same work takes 8 times longer,
    # and a charger going in is exactly when the governor is being reshuffled.
    #
    # So the world gives up its remaining steps rather than the frame: past
    # this it simply runs slow, which looks like the music easing off for a
    # moment, where a stalled panel looks broken. What is given up is not
    # owed, either, or the next frame would inherit the debt and stall in its
    # turn.
    BUDGET = 0.008

    # Surface. Cells are 1/SCALE of a pixel; speeds are in cells per second.
    WAVE_SPEED = 26.0           # ripples cross the panel's width in ~1 s
    WAVE_DAMPING = 2.2          # 1/s; how quickly ringing dies away
    WAVE_SPRING = 6.0           # 1/s²; pulls the surface back to level
    DRIVE = 14.0                # push per unit of band loudness
    KICK = 5.0                  # velocity per unit of onset

    # Glow.
    GLOW_DECAY = 4.5            # 1/s
    GLOW_DIFFUSION = 5.0        # cells²/s
    GLOW_RISE = 5.0             # cells/s upward
    GLOW_DRIFT = 5.0            # cells/s per unit of surface slope
    GLOW_FEED = 0.9             # per second, per unit of band loudness
    # Flashes: light a strike gives off at once, which spreads fast and is
    # gone in a blink, where glow lingers.
    FLASH = 1.1                 # per unit of onset
    FLASH_DECAY = 13.0          # 1/s
    FLASH_SPREAD = 30.0         # cells²/s
    # A strike only flashes by as much as its onset clears FLASH_THRESHOLD,
    # and a band has a charge to flash with, like a capacitor or a nerve:
    # flashing spends it and it builds back over FLASH_RECHARGE. A band
    # struck on every beat flashes on the first and glimmers after, so the
    # flashes that do come mean something, rather than whiting out the panel
    # on every hit of a dense mix.
    FLASH_THRESHOLD = 0.12
    FLASH_RECHARGE = 1.5        # seconds
    FLASH_DRAIN = 2.5           # charge spent per unit of flash

    # Air. A strike is a pressure pulse, with the energy of one: its strength
    # squared - how sharp the onset is - times the size of what struck - the
    # band's footprint, so a bass kick moves hundreds of times the air a
    # hi-hat does. Each band's pulse drives a spring that shoves the picture
    # around it outward and snaps it back past rest before settling, all
    # within a fraction of a second.
    #
    # A pulse on the surface spreads in two dimensions, so how far a point is
    # shoved goes as the energy over its distance, up to a share of that
    # distance it approaches smoothly and cannot pass. So there is no
    # threshold: a weak pulse's saturated zone - growing as the square root
    # of the energy, as a blast's on a surface does - is under a pixel, and
    # past it the shove is too small to see; a strong one saturates across
    # pixels and still shoves what is well beyond. Small stuff all but
    # vanishes and big hits slam, from the one rule.
    #
    # How fast the spring snaps back follows how the strike arrived. A band
    # whose whole rise came in its strike - a punchy kick - springs at
    # BLAST_FREQ; one that had mostly swelled up before it struck springs at
    # BLAST_SOFT_FREQ, and anything between, between. The sound is heard 43
    # ms at a time, so this tells punchy from swelling, not a click from a
    # punch.
    #
    # And a strike is not all of it: for as long as the band stays louder than
    # its recent level, that pressure goes on pushing - its energy worked out
    # the same way, times BLAST_HOLD - so a boom that rings holds the push out
    # while it rings and lets go as it fades, and a tight kick is only its
    # strike. Measured against the band's own recent level, a steady bassline
    # holds nothing out for long.
    # Both frequencies are set so a push peaks outward when the rise that
    # struck it would have: with this damping, a spring kicked from rest peaks
    # at 0.197 / f seconds. The sharpest rise the hearing can tell apart is
    # about 21 ms, half its 43 ms window, which is 9.4 Hz - a push back
    # through rest by 60 ms, about the punch of a kick drum, and out for more
    # than a frame, so the push itself is seen and not only the return. The
    # slowest rise that still strikes rises faster than ONSET_RATE, so for a
    # typical rise of half the range, over 100 ms: 2 Hz. Sharpness is about a
    # frame over the rise time, so blending by it in frequency, not period,
    # keeps the peak on the rise between the two as well.
    BLAST_FREQ = 9.4            # Hz, for the sharpest strikes: peaks at 21 ms
    BLAST_SOFT_FREQ = 2.0       # Hz, for strikes that swelled up first: peaks at 100 ms
    BLAST_HOLD = 0.2            # sustained push, per unit of energy held; set by eye
    BLAST_STRUCK_TAU = 0.3      # seconds a strike counts against retuning by the next
    BLAST_DAMPING = 0.45        # of critical; enough to overshoot once, inward
    BLAST_STRENGTH = 6.0        # shove per unit of energy, over the distance; set by eye
    BLAST_CORE = 0.8            # cells of saturated zone per root of energy
    # The most a push moves anything, as a share of its distance from the
    # pulse. Falling off as the inverse of the distance, a shove can pull
    # points apart or together by at most a third of this, so anything under
    # one never folds the picture over itself.
    BLAST_MAX = 0.8
    BLAST_UNSEEN = 0.02         # energy under which a pulse moves nothing to see

    # Sparks.
    MAX_SPARKS = 90
    SPARK_RATE = 40.0           # per second, per unit of band loudness
    SPARK_BURST = 18.0          # per unit of onset
    SPARK_LIFE = (0.15, 0.7)    # seconds
    SPARK_SURF = 5.0            # acceleration per unit of surface slope
    SPARK_LIFT = 10.0           # cells/s² upward
    SPARK_DRAG = 3.0            # 1/s
    SPARK_LIGHT = 0.45

    # Light. Each band is a lamp under the surface where it sits; its light is
    # bent by the slope of the water it passes through and lands elsewhere.
    SUBRAYS = 3                 # rays per cell, each way
    REFRACTION = 14.0           # cells a ray is bent per unit of slope
    LAMP = 0.9
    # The widest a lamp's light reaches, as a share of the width. The bass's
    # footprint grows with the world's height, and grown over the whole panel
    # its lamp lit nearly all of it white on every kick; capped, the bass
    # still heaves the whole surface, but its light stays a patch within it.
    LAMP_REACH = 1 / 3
    # Sunlight: even light from above, as bright as the music is loud, bent
    # through the water onto a floor SUN_DEPTH below - deeper than the lamps,
    # so the same ripples bend it further and it gathers into the sharp,
    # wriggling network of lines on the floor of a pool. Only light gathered
    # past SUN_FOCUS times what flat water sends shows, so the lines are drawn
    # over the dark rather than over a grey wash.
    SUN_DEPTH = 65.0            # cells a sun ray is bent per unit of slope
    SUN = 0.35
    SUN_FOCUS = 1.25
    SUN_TAU = 0.3               # seconds the sun takes to follow the music
    # A caustic line is thinner than a pixel, and averaging it into one would
    # lose it; a pixel it crosses is lit mostly by the brightest of its cells.
    SUN_SHARPNESS = 0.7
    LAMP_TAU = 0.05             # seconds a lamp takes to follow its band
    SETTLE_TAU = 1.5            # seconds the light takes to get used to a level
    SETTLE = 0.8                # how much of a steady level it stops showing
    STEADY_SHINE = 0.25         # how much of the level itself still shines
    # Raised water brightens the light that is already there, by this much
    # per unit of height above level, rather than giving off light of its
    # own. Added as light, a bass heave lifted the whole surface into a grey
    # wall that never cleared between beats; as a gain, the picture pumps
    # with the bass and the dark between things stays dark.
    SWELL = 0.3
    GAMMA = 2.4                 # the panel is linear; the eye is not
    # Light under this fraction of the adapted range is let fall to black, so
    # the dark between things stays dark instead of a haze of dim pixels.
    TOE = 0.22
    # Like an eye's: the brightest of the picture is kept near ADAPT_TARGET
    # after the tone curve, closing down within a fraction of a second and
    # opening back up over a couple, so a flash still flares past it before
    # it adapts. Never opens past MAX_EXPOSURE, so faint sound stays faint.
    ADAPT_TARGET = 3.0
    ADAPT_CLOSE, ADAPT_OPEN = 0.25, 2.5     # seconds
    MIN_EXPOSURE, MAX_EXPOSURE = 0.15, 3.0
    ADAPT_PERCENTILE = 96       # the highlights, not the average, set it
    COVERAGE_WEIGHT = 2.6       # ...unless the average is this close to them

    # The world's proportions were found at 5 rows. Taller, footprints grow
    # with the square root of the height, so bass still fills a full panel
    # without every band blurring into one.
    TUNED_ROWS = 5

    def __init__(self, width, height, bands, seed=7):
        self.width = width
        self.w = width * self.SCALE
        self.bands = bands
        self._rng = np.random.default_rng(seed)

        # How each band acts, from its place in the spectrum, 0 low to 1 high.
        self._pitch = pitch = np.linspace(0.0, 1.0, bands)
        self._base_reach = 8.0 * (0.08 / 8.0) ** pitch + 0.9  # footprint, cells
        self._rate = 2 * math.pi * (0.35 * (10.0 / 0.35) ** pitch)   # rad/s
        self._warmth = (1.0 - pitch) ** 1.6                   # glow left
        self._brightness = np.clip((pitch - 0.45) / 0.55, 0, 1) ** 1.3  # sparks
        self._wander_freq = self._rng.uniform(0.08, 0.35, (bands, 2))
        self._wander_phase = self._rng.uniform(0, 2 * math.pi, (bands, 2))
        self._sharpness = (self._base_reach.max() / self._base_reach) ** 0.6
        self._shape(height * self.SCALE)
        self.reset()

    def _shape(self, cells):
        """Size the world to `cells` tall, and everything laid out by height."""
        self.h = cells
        self.height = -(-cells // self.SCALE)   # panel rows, the last maybe partial
        self._ys, self._xs = np.mgrid[0:self.h, 0:self.w].astype(np.float64)
        scale = math.sqrt(cells / (self.TUNED_ROWS * self.SCALE))
        self._reach = self._base_reach * scale
        self._depth = self.h * (0.78 - 0.5 * self._pitch)     # resting height
        # Rays of light through the surface, SUBRAYS each way per cell.
        n = self.SUBRAYS
        ry, rx = np.mgrid[0:self.h * n, 0:self.w * n].astype(np.float64)
        self._ray_x = ((rx + 0.5) / n - 0.5).reshape(-1).clip(0, self.w - 1.001)
        self._ray_y = ((ry + 0.5) / n - 0.5).reshape(-1).clip(0, self.h - 1.001)
        # The rays start from the same places every frame, so how to look up
        # a field at them is worked out once.
        x0, y0 = self._ray_x.astype(int), self._ray_y.astype(int)
        fx, fy = self._ray_x - x0, self._ray_y - y0
        base = y0 * self.w + x0
        self._ray_lookup = (
            np.stack((base, base + 1, base + self.w, base + self.w + 1)),
            np.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)))

    def resize(self, cells):
        """Make the world `cells` tall, keeping what is in it.

        Growing opens new, still water below what is there, which the waves
        above spill into, and the bands' resting places sink with the floor;
        shrinking lets go of whatever was below the new edge. Either way
        nothing is restarted, so a view of the world can stretch and shrink
        smoothly around it.
        """
        cells = max(int(cells), self.SCALE)
        if cells == self.h:
            return
        old = self.h
        def fit(field):
            if cells < old:
                return field[:cells].copy()
            return np.vstack((field, np.zeros((cells - old, self.w))))
        self.height_field = fit(self.height_field)
        self.velocity = fit(self.velocity)
        self.glow = fit(self.glow)
        self.flash = fit(self.flash)
        self._lamps = fit(self._lamps)
        self._shape(cells)
        self._place()

    def reset(self):
        self.height_field = np.zeros((self.h, self.w))
        self.velocity = np.zeros((self.h, self.w))
        self.glow = np.zeros((self.h, self.w))
        self.flash = np.zeros((self.h, self.w))
        self._phase = self._rng.uniform(0, 2 * math.pi, self.bands)
        self._roam = np.zeros(self.bands)         # each band's wandering clock
        self._balance = np.zeros(self.bands)
        self._sparks = np.zeros((0, 6))           # x, y, vx, vy, age, life
        self._spark_light = np.zeros(0)
        self._carry = 0.0
        self._strike = np.zeros(self.bands)       # onsets not yet struck
        self._charge = np.ones(self.bands)        # what each band can flash with
        self._blast = np.zeros(self.bands)        # how far each band pushes out
        self._blast_speed = np.zeros(self.bands)
        self._blast_sharp = np.ones(self.bands)   # 1 a punchy last strike, 0 a swell
        self._blast_struck = np.zeros(self.bands)  # energy of recent strikes, fading
        self._place()
        self._exposure = self.MAX_EXPOSURE
        self._lamps = np.zeros((self.h, self.w))
        self._settled = np.zeros(self.bands)     # each band's recent level
        self._sun = 0.0
        self._adapt_dt = 0.0

    # ── Physics ──────────────────────────────────────────────────────────────

    @staticmethod
    def _neighbours(field):
        """Each cell's neighbour above, below, left and right, the edges
        standing in for what is past them - so the edges reflect."""
        up = np.concatenate((field[:1], field[:-1]), axis=0)
        down = np.concatenate((field[1:], field[-1:]), axis=0)
        left = np.concatenate((field[:, :1], field[:, :-1]), axis=1)
        right = np.concatenate((field[:, 1:], field[:, -1:]), axis=1)
        return up, down, left, right

    @classmethod
    def _laplacian(cls, field):
        up, down, left, right = cls._neighbours(field)
        return up + down + left + right - 4 * field

    @classmethod
    def _gradient(cls, field):
        up, down, left, right = cls._neighbours(field)
        return (right - left) / 2, (down - up) / 2

    def _sample(self, field, x, y):
        """Bilinear lookup at fractional cells, held at the edges."""
        x = np.clip(x, 0, self.w - 1.001)
        y = np.clip(y, 0, self.h - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        return ((field[y0, x0] * (1 - fx) + field[y0, x0 + 1] * fx) * (1 - fy)
                + (field[y0 + 1, x0] * (1 - fx) + field[y0 + 1, x0 + 1] * fx) * fy)

    def _sources(self):
        """Where each band is in the world, as (x, y) arrays of cells."""
        centre = (self.w - 1) / 2
        across = centre * (1 + 0.85 * self._balance)
        wander = np.sin(self._roam[:, None] * self._wander_freq * 2 * math.pi
                        + self._wander_phase)
        x = across + wander[:, 0] * self.w * 0.35
        y = self._depth + wander[:, 1] * self.h * 0.22
        return np.clip(x, 0, self.w - 1), np.clip(y, 0, self.h - 1)

    def _place(self):
        """Where every band is this frame, and the gaussian footprints it
        pushes, strikes and flashes with, each bands x h x w with a peak of 1.
        Once a frame rather than once a step: the sources drift far too slowly
        for the difference to show."""
        sx, sy = self._sources()
        dx = self._xs[None] - sx[:, None, None]
        dy = self._ys[None] - sy[:, None, None]
        distance = -(dx * dx + dy * dy) / 2
        spread = lambda reach: np.exp(distance / (reach[:, None, None] ** 2))
        # The strike is spread a little wider than the push, so even a high
        # band's strike is big enough to ring; the flash wider again.
        self._at = (sx, sy)
        self._feet = spread(self._reach)
        self._lamp_feet = spread(np.minimum(self._reach, self.LAMP_REACH * self.w))
        self._strike_feet = spread(np.maximum(self._reach, 1.6))
        self._flash_feet = spread(self._reach * 1.2 + 1.0)

    def _step(self, dt, loud, onset):
        self._phase += self._rate * dt
        self._roam += dt * (0.25 + 2.5 * loud)
        sx, sy = self._at
        feet = self._feet
        struck = onset.any()

        # The lamps: every band shines where it is, as brightly as it is loud.
        # Like a sense, the light gets used to a band that stays loud: what it
        # shows is mostly how far each band stands out from its own recent
        # level, with a little of the level itself. A dense, compressed mix
        # would otherwise light every lamp at once, and adapting to that
        # leaves a flat grey panel; this way the water still churns under the
        # whole of it while the light picks out what is changing.
        self._settled += (loud - self._settled) * (1 - math.exp(-dt / self.SETTLE_TAU))
        stand_out = np.clip(loud - self.SETTLE * self._settled, 0, 1)
        shine = self.STEADY_SHINE * loud ** 2 + stand_out ** 1.2
        lamps = (shine[:, None, None] * self._lamp_feet).sum(axis=0)
        self._sun += (np.mean(loud ** 1.5) - self._sun) * (1 - math.exp(-dt / self.SUN_TAU))
        self._lamps += (lamps - self._lamps) * (1 - math.exp(-dt / self.LAMP_TAU))

        # The surface: pushed by every band, oscillating at its own rate, and
        # struck by onsets. A narrow push is a stronger one, as a fingertip
        # dents what a palm only presses, so a high band's ripples are as
        # sharp as a low band's swells are broad.
        push = (self.DRIVE * loud ** 1.5 * np.sin(self._phase)
                * self._sharpness)[:, None, None]
        force = (push * feet).sum(axis=0)
        h, v = self.height_field, self.velocity
        v += dt * (self.WAVE_SPEED ** 2 * self._laplacian(h)
                   - self.WAVE_DAMPING * v - self.WAVE_SPRING * h + force)
        if struck:
            kick = self.KICK * onset * np.sqrt(self._sharpness)
            v += (kick[:, None, None] * self._strike_feet).sum(axis=0)
        h += dt * v

        # Glow: fed by warm bands and flashed by onsets, then carried by the
        # currents across the surface and upward, spread and cooled.
        feed = (self.GLOW_FEED * dt * self._warmth * shine)[:, None, None] * feet
        g = self.glow + feed.sum(axis=0)
        f = self.flash
        self._charge += (1 - self._charge) * (1 - math.exp(-dt / self.FLASH_RECHARGE))
        if struck:
            sharp = np.maximum(onset - self.FLASH_THRESHOLD, 0.0)
            spent = np.minimum(self._charge, sharp * self.FLASH_DRAIN)
            burst = self.FLASH * sharp * self._charge * (0.35 + self._warmth)
            self._charge -= spent
            f = f + (burst[:, None, None] * self._flash_feet).sum(axis=0)
        if f.max() > 1e-4:
            # Spread in two half steps, which keeps diffusion this fast stable.
            for _ in range(2):
                f = f + self.FLASH_SPREAD * dt / 2 * self._laplacian(f)
            f = f * math.exp(-self.FLASH_DECAY * dt)
        self.flash = f
        gx, gy = self._gradient(h)
        back_x = self._xs + self.GLOW_DRIFT * gx * dt
        back_y = self._ys + (self.GLOW_DRIFT * gy + self.GLOW_RISE) * dt
        g = self._sample(g, back_x, back_y)
        g += self.GLOW_DIFFUSION * dt * self._laplacian(g)
        self.glow = g * math.exp(-self.GLOW_DECAY * dt)

        # Air: a spring per band, kicked outward by the energy of its strikes
        # and held out by the energy of loudness that stays up.
        above = np.maximum(loud - self._settled, 0.0)
        self._blast_struck *= math.exp(-dt / self.BLAST_STRUCK_TAU)
        if struck:
            energy = onset ** 2 * self._reach ** 2
            # How much of the band's rise came in this strike, rather than
            # before it; weighted by the strike's share of recent strikes, so
            # a faint one does not retune a big push in flight.
            sharp = np.clip(onset / np.maximum(np.maximum(above, onset), 1e-6), 0.0, 1.0)
            weight = energy / (energy + self._blast_struck + 1e-9)
            self._blast_sharp += (sharp - self._blast_sharp) * weight
            self._blast_struck += energy
        soft = 2 * math.pi * self.BLAST_SOFT_FREQ
        omega = soft + (2 * math.pi * self.BLAST_FREQ - soft) * self._blast_sharp
        if struck:
            self._blast_speed += omega * energy
        # Pushing as omega squared, the push held out does not depend on the
        # spring's speed, only on the pressure.
        held = omega * omega * self.BLAST_HOLD * above ** 2 * self._reach ** 2
        self._blast_speed += dt * (held - omega * omega * self._blast
                                   - 2 * self.BLAST_DAMPING * omega * self._blast_speed)
        self._blast += dt * self._blast_speed

        self._move_sparks(dt, gx, gy)
        self._throw_sparks(dt, loud, onset, sx, sy)

    def _throw_sparks(self, dt, loud, onset, sx, sy):
        rate = self._brightness * (self.SPARK_RATE * loud ** 2 * dt
                                   + self.SPARK_BURST * onset)
        counts = self._rng.poisson(rate)
        room = self.MAX_SPARKS - len(self._sparks)
        if room <= 0 or counts.sum() == 0:
            return
        which = np.repeat(np.arange(self.bands), counts)[:room]
        n = len(which)
        spread = self._reach[which] + 1.0
        angle = self._rng.uniform(0, 2 * math.pi, n)
        speed = self._rng.uniform(4, 30, n) * (0.4 + onset[which] * 3)
        life = self._rng.uniform(*self.SPARK_LIFE, n)
        new = np.column_stack((
            sx[which] + self._rng.normal(0, spread),
            sy[which] + self._rng.normal(0, spread),
            np.cos(angle) * speed, np.sin(angle) * speed,
            np.zeros(n), life))
        light = 0.5 + loud[which] + onset[which] * 2
        self._sparks = np.vstack((self._sparks, new))
        self._spark_light = np.concatenate((self._spark_light, light))

    def _move_sparks(self, dt, gx, gy):
        s = self._sparks
        if not len(s):
            return
        # Surfing: pushed down the slope of the surface where each one is.
        s[:, 2] += dt * (-self.SPARK_SURF * self._sample(gx, s[:, 0], s[:, 1]) * 60
                         - self.SPARK_DRAG * s[:, 2])
        s[:, 3] += dt * (-self.SPARK_SURF * self._sample(gy, s[:, 0], s[:, 1]) * 60
                         - self.SPARK_DRAG * s[:, 3] - self.SPARK_LIFT)
        s[:, 0] += dt * s[:, 2]
        s[:, 1] += dt * s[:, 3]
        s[:, 4] += dt
        alive = ((s[:, 4] < s[:, 5]) & (s[:, 0] > -1) & (s[:, 0] < self.w)
                 & (s[:, 1] > -1) & (s[:, 1] < self.h))
        self._sparks, self._spark_light = s[alive], self._spark_light[alive]

    # ── Light ────────────────────────────────────────────────────────────────

    def _spark_field(self):
        field = np.zeros((self.h, self.w))
        s = self._sparks
        if not len(s):
            return field
        age = s[:, 4] / s[:, 5]
        # Flares up fast and burns out, flickering as it goes.
        envelope = np.minimum(age * 8, 1) * (1 - age) ** 1.5
        flicker = 0.65 + 0.35 * np.sin(s[:, 4] * 55 + s[:, 5] * 97)
        light = self.SPARK_LIGHT * self._spark_light * envelope * flicker
        light *= self.SCALE * self.SCALE     # a spark is a point, not a cell
        x = np.clip(s[:, 0], 0, self.w - 1.001)
        y = np.clip(s[:, 1], 0, self.h - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        for dx, dy, weight in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                               (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            np.add.at(field, (y0 + dy, x0 + dx), light * weight)
        return field

    def _refracted(self):
        """The lamps' light and the sunlight, after the surface has bent them.

        Every ray leaves its cell carrying that cell's share of lamplight, is
        turned by the slope of the water there, and lands where it lands. Where
        the surface curves like a lens rays crowd together into bright
        filaments, and where it curves the other way they spread and leave it
        dark - caustics, as on the floor of a pool, dancing as the water does.

        Returns (lamplight, focus): what the lamps' light comes to at each
        cell, and how many times more rays than flat water would send each
        cell - 1 everywhere under a still surface.
        """
        gx, gy = self._gradient(self.height_field)
        index, weight = self._ray_lookup
        at_rays = lambda field: (field.reshape(-1)[index] * weight).sum(axis=0)
        slope_x, slope_y = at_rays(gx), at_rays(gy)
        carried = at_rays(self._lamps) * (self.LAMP / (self.SUBRAYS * self.SUBRAYS))
        light, _ = self._land(slope_x, slope_y, self.REFRACTION, carried)
        _, focus = self._land(slope_x, slope_y, self.SUN_DEPTH, None)
        return light, focus

    def _land(self, slope_x, slope_y, depth, carried):
        """Bend every ray by the slope under it, as far as `depth` makes it,
        and add up what lands in each cell: the light `carried`, if given, and
        how many rays, as a multiple of what flat water would send."""
        x = self._ray_x - depth * slope_x
        y = self._ray_y - depth * slope_y
        # Rays that would leave the panel bounce back off its edges.
        x = np.abs(x)
        x = np.where(x > self.w - 1, 2 * (self.w - 1) - x, x).clip(0, self.w - 1.001)
        y = np.abs(y)
        y = np.where(y > self.h - 1, 2 * (self.h - 1) - y, y).clip(0, self.h - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0
        index = y0 * self.w + x0
        size = self.h * self.w
        corners = ((index, (1 - fx) * (1 - fy)), (index + 1, fx * (1 - fy)),
                   (index + self.w, (1 - fx) * fy), (index + self.w + 1, fx * fy))
        if carried is not None:
            light = sum(np.bincount(i, carried * w, size) for i, w in corners)
            return light.reshape(self.h, self.w), None
        rays = sum(np.bincount(i, w, size) for i, w in corners)
        return None, (rays / (self.SUBRAYS * self.SUBRAYS)).reshape(self.h, self.w)

    def _blown(self, field):
        """The field as the air has it this instant: each point moved out
        from every band by as far as that band's pulse shoves it.

        Worked backwards, as the glow's currents are: each cell shows what is
        at the place the push moved to it from. What comes in from past an
        edge is the picture reflected in it.
        """
        blast = self._blast
        pushing = np.abs(blast) > self.BLAST_UNSEEN
        if not pushing.any():
            return field
        blast = blast[pushing, None, None]
        sx, sy = self._at
        dx = self._xs[None] - sx[pushing, None, None]
        dy = self._ys[None] - sy[pushing, None, None]
        # Inside the saturated zone, which grows as the root of the energy,
        # everything is shoved as far as it can go; outside, by the energy
        # over the distance. As a share of the distance, that is over its
        # square.
        core = self.BLAST_CORE * np.sqrt(np.abs(blast))
        shove = self.BLAST_STRENGTH * blast / (dx * dx + dy * dy + core * core)
        share = self.BLAST_MAX * np.tanh(shove / self.BLAST_MAX)
        # Overlapping pushes add, but together still never fold.
        total = share.sum(axis=0)
        limit = np.maximum(np.abs(total) / self.BLAST_MAX, 1.0)
        x = self._xs - (share * dx).sum(axis=0) / limit
        y = self._ys - (share * dy).sum(axis=0) / limit
        x = np.abs(x)
        x = np.where(x > self.w - 1, 2 * (self.w - 1) - x, x)
        y = np.abs(y)
        y = np.where(y > self.h - 1, 2 * (self.h - 1) - y, y)
        return self._sample(field, x, y)

    def light(self):
        """The world as the panel shows it: height x width, 0 to 255."""
        h = self.height_field
        swell = 1 + np.maximum(h, 0) * self.SWELL
        lamplight, focus = self._refracted()
        # The air moves what drifts - glow, sparks, the lines on the floor -
        # but not the lamps and flashes that are the thumps themselves: blown
        # outward from under a thump, those only swelled it bigger and whiter.
        drifting = self._blown(self.glow + self._spark_field())
        lit = (drifting + self.flash + lamplight) * swell
        caustic = self._blown(self.SUN * self._sun * np.maximum(focus - self.SUN_FOCUS, 0))
        s = self.SCALE
        if self.h % s:
            # A last row only partly grown shows as dim as it is part there.
            pad = np.zeros((s - self.h % s, self.w))
            lit, caustic = np.vstack((lit, pad)), np.vstack((caustic, pad))
        blocks = lambda field: field.reshape(self.height, s, self.width, s)
        pixels = blocks(lit).mean(axis=(1, 3))
        lines = blocks(caustic)
        pixels = pixels + ((1 - self.SUN_SHARPNESS) * lines.mean(axis=(1, 3))
                           + self.SUN_SHARPNESS * lines.max(axis=(1, 3)))
        # Keyed to the highlights, but also to how much of the panel is lit:
        # a few bright things can be white, while light spread over most of
        # the panel is held to a grey rather than whiting it all out.
        bright = max(np.percentile(pixels, self.ADAPT_PERCENTILE),
                     self.COVERAGE_WEIGHT * pixels.mean())
        if bright > 1e-3:
            want = min(max(self.ADAPT_TARGET / bright, self.MIN_EXPOSURE),
                       self.MAX_EXPOSURE)
            tau = self.ADAPT_CLOSE if want < self._exposure else self.ADAPT_OPEN
            self._exposure += (want - self._exposure) * (1 - math.exp(-self._adapt_dt / tau))
        self._adapt_dt = 0.0
        value = 1 - np.exp(-self._exposure * pixels)
        value = np.maximum(value - self.TOE, 0) / (1 - self.TOE)
        value = value ** self.GAMMA
        return np.rint(value * 255).astype(np.uint8)

    def advance(self, dt, loud, onset, balance):
        """Run the world on by dt seconds under the sound as it is now, or by
        as much of it as BUDGET allows.

        `loud` and `onset` are per band, 0 to 1ish; `balance` is per band,
        -1 all left to 1 all right.
        """
        self._balance += (balance - self._balance) * (1 - math.exp(-dt / 0.15))
        # A strike lands once, on the next step, however the frames and steps
        # happen to fall.
        self._strike = np.maximum(self._strike, onset)
        self._adapt_dt += dt
        self._carry += dt
        self._place()
        deadline = time.monotonic() + self.BUDGET
        while self._carry >= self.STEP:
            self._carry -= self.STEP
            self._step(self.STEP, loud, self._strike)
            self._strike = np.zeros(self.bands)
            if time.monotonic() >= deadline:
                self._carry = 0.0       # given up, not owed
                break
