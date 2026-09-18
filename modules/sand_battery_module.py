import json
import logging
import math
import os
import random
import threading
import time

import numpy as np
from PIL import Image

from modules.module_base import ModuleBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PEAK_FILE = os.path.join(BASE_DIR, 'power_peaks.json')


_readers = {}


def _battery_reader(path, interval, nominal):
    """The one reader for a battery, however many gauges ask for it.

    Shared, and outliving the modules: a config reload builds fresh modules,
    and a thread apiece would pile up one per swap.
    """
    reader = _readers.get(path)
    if reader is None:
        reader = _readers[path] = _BatteryReader(path, interval, nominal)
    return reader


class _BatteryReader:
    """What the battery says, read away from the thread that draws.

    Reading /sys/class/power_supply goes through ACPI to the embedded
    controller, and the controller has other things to do for a while after a
    charger goes in: reads that take a third of a millisecond when it is idle
    were measured at up to 33 ms then, and from inside the service whole
    frames went into them - up to 650 ms at a time, which the panel showed as
    a stutter a moment after plugging in, every time. The cost was never the
    CPU it takes to read sysfs, which is nothing; it is that the read waits on
    hardware, and nothing between two frames may wait on hardware. So this
    reads on its own thread, and the gauge draws with whatever came back last.

    The first reading is taken before anything draws, so the gauge has a
    battery to prime its pile from rather than an empty one.
    """

    def __init__(self, path, interval, nominal):
        self.path = path
        self.interval = interval
        self.nominal = nominal
        self.reading = self._take()     # (when, status, capacity, watts)
        threading.Thread(target=self._loop, daemon=True).start()

    def _read(self, name, cast=int):
        try:
            with open(f'{self.path}/{name}', 'r') as f:
                return cast(f.read().strip())
        except (OSError, ValueError):
            return None

    def _power(self):
        """Watts flowing in or out, from the battery's own current and voltage.

        Power, not current: it is the quantity that actually varies with load,
        and it stays comparable between charging and discharging even though
        pack voltage differs between the two.
        """
        amps = abs(self._read('current_now') or 0) / 1e6
        # Fall back through nominal pack voltage rather than treating a missing
        # reading as 0 V, which would silently mean "never draw any flow".
        micro_volts = (self._read('voltage_now')
                       or self._read('voltage_min_design')
                       or self.nominal)
        return amps * (micro_volts / 1e6)

    def _take(self):
        # The clock is read first, so a reading is stamped with when it was
        # asked for rather than with however long the controller took over it.
        now = time.monotonic()
        return (now, self._read('status', str) or 'Unknown',
                self._read('capacity'), self._power())

    def _loop(self):
        while True:
            time.sleep(self.interval)
            self.reading = self._take()


class SandBatteryModule(ModuleBase):
    """Battery gauge drawn as a falling-sand pile.

    Stands in for the PrecomputedNoise scroll band plus the 1px BatteryModule
    bar, which together occupy rows 13-18. That is 9x6 = exactly 54 cells, so
    one speck represents 1/54th of charge and 100% lights every cell.

    The panel is mounted sideways, so gravity points toward -x (image left)
    and the six lanes the sand falls down are panel *rows*, indexed by y. Each
    lane is 9 cells deep, and settled sand leaves no gaps, so a lane is fully
    described by how many grains have come to rest in it.

    The lane distribution only chooses where a speck is *released*. Where it
    ends up is decided by the sand: a grain rolls downhill on landing, and any
    slope steeper than `repose` collapses over time, so the pile relaxes into a
    rounded heap instead of standing up as a histogram of the drop weights.

    Flow is deliberately over-drawn: far more specks stream in or out than the
    net change in charge calls for, because the stream is what shows current
    direction and magnitude. Only the specks needed to match the battery
    percentage stay; the excess runs down the surface of the pile and is gone,
    off a side edge if the slope leads there, otherwise resting in the first
    dip it reaches. It is never sent uphill to find an edge - a grain that
    climbs reads as a bug even to someone who could not say why.

    Discharge has two ways out, because one is not legible across the whole
    range. A white speck lifting off the surface is clear against the dark
    space above a low pile and invisible against a nearly full one, so the
    other way is a bubble: a black void that starts at the floor, climbs up
    through the sand, and bursts at the surface into exactly that white speck.
    Which one a departing speck uses is decided by the fill, so the panel is
    reading floaters at empty and bubbles at full, and at no charge does a
    discharge look the same as sitting still.

    Sand and flow are drawn differently. A grain that is going to settle, or
    one really leaving the pile, is a solid speck of the same brightness as
    the sand, because it is the charge itself. Everything cosmetic - the surplus
    that pours in and runs off at a full-ish battery, and the specks and
    bubbles that keep leaving without the pile shrinking - is made of
    translucent particles instead, each carrying a fraction of a speck's light
    and several times as many of them. Where they overlap their light adds up,
    so a heavy stream reads as a continuous shimmer of varying brightness
    rather than as a few hard dots, and they move in fractions of a cell, their
    light shared between the two cells either side of where they really are.
    The amount of light flowing for a given wattage is unchanged; only how
    finely it is divided.

    Bubbles are the exception: always solid, a whole cell fully dark. The
    panel's brightness is linear PWM, and at the top of its range a cell
    dimmed by a third of a speck, or a whole one shared between two cells,
    barely looks different from the sand around it. A translucent bubble is
    instead spawned with its alpha as the odds, so fewer, clearer holes carry
    the same darkness on average.
    """

    SYS = '/sys/class/power_supply/BAT1'
    IDLE_RATE = 8.0       # specks/sec used only to correct drift when idle
    MAX_DT = 0.25         # clamp, so a stall can't teleport the whole pile
    NOMINAL_MICROVOLTS = 15_480_000   # 4S pack nominal, last-resort fallback

    # Full scale is learned from what this machine has actually drawn, so the
    # brightest snow means "as hard as you have ever pushed it" rather than a
    # number guessed off a spec sheet.
    FEASIBLE_WATTS = 200.0    # beyond this it is a bad reading, not a real load
    SAMPLE_INTERVAL = 0.5     # seconds between battery reads
    PEAK_CONFIRM = 3          # samples a candidate peak must hold for (~1.5s)
    PEAK_WRITE_INTERVAL = 60.0  # don't rewrite the peaks file more often

    def __init__(self, height=6, spread=1.1, fall_speed=11.0,
                 roll_speed=9.0, float_speed=7.0, fizz_speed=6.0,
                 sway_width=(0.7, 2.2), sway_rate=(6.0, 12.0), float_jitter=0.25,
                 repose=1, slump_rate=30.0,
                 floor_watts=45.0, max_density=0.9, fizz_density=0.3,
                 particle_alpha=0.35, peak_file=None):
        super().__init__(height)
        self.lanes = height
        self.depth = None          # cells per lane; set on first render
        self.heights = [0] * self.lanes
        self.falling = []          # specks dropping in from offscreen
        self.rolling = []          # surplus running downhill across the surface
        self.floating = []         # specks leaving the pile on discharge
        self.bubbles = []          # voids rising through the pile, to burst as one

        self.fall_speed = fall_speed
        self.roll_speed = roll_speed
        self.float_speed = float_speed
        self.fizz_speed = fizz_speed
        # Specks leaving the pile flutter rather than tracking a fixed diagonal.
        # Width and rate are drawn per speck: a shared frequency reads as one
        # rippling wave instead of as snow, and a shared phase even more so.
        self.sway_width = tuple(sway_width)   # lanes, peak deviation
        self.sway_rate = tuple(sway_rate)     # radians/sec
        self.float_jitter = float_jitter      # +/- share of outward speed
        # Full scale is the hardest this machine has been seen to push, tracked
        # per direction: a charge peaks well below a full-load discharge, so
        # sharing one scale would leave charging permanently unable to reach
        # full snow. floor_watts stops a history of nothing but idling from
        # rescaling a 5W trickle into a blizzard.
        self.floor_watts = floor_watts
        # Fraction of the *empty* cells carrying specks at full scale. Below 1
        # on purpose: the pile's edge has to stay findable through the snow.
        self.max_density = max_density
        # The same for the *filled* cells, and far lower. A bubble unlights
        # sand, so this is the share of the pile allowed to be holes at full
        # scale: enough to fizz, not enough to stop the bar reading as a bar.
        self.fizz_density = fizz_density
        # Share of a speck's light each translucent flow particle carries, so
        # also how many more of them there are: 0.35 is about three per speck.
        self.particle_alpha = particle_alpha
        self.peak_file = DEFAULT_PEAK_FILE if peak_file is None else peak_file
        self.peaks = {'charge': 0.0, 'discharge': 0.0}
        self._load_peaks()
        self._streak = 0            # consecutive samples above the stored peak
        self._streak_key = None
        self._streak_floor = 0.0
        self._peaks_written = None

        # Battery state, as the reader last had it. On a cadence rather than
        # every frame because confirming a peak over 3 samples only rejects
        # spikes if those samples span a meaningful stretch of time - and on
        # its own thread because the reads wait on the embedded controller.
        self._battery = _battery_reader(self.SYS, self.SAMPLE_INTERVAL,
                                        self.NOMINAL_MICROVOLTS)
        self._sampled_at = None
        self._status = 'Unknown'
        self._capacity = None
        self._watts = 0.0
        # Angle of repose, in cells of depth per lane. 1 is a 45 degree slope,
        # the steepest a falling-sand grain can hold when its only options are
        # straight down or diagonally down.
        self.repose = repose
        self.slump_rate = slump_rate   # grains/sec that avalanche

        # Where specks are *released* is biased hard toward the middle lanes,
        # which is what keeps the heap centred; the pile shape itself comes
        # from the sand settling, not from these weights.
        centre = (self.lanes - 1) / 2.0
        weights = np.exp(-(((np.arange(self.lanes) - centre) / spread) ** 2))
        self.lane_weights = weights / weights.sum()

        self._credit = 0.0         # fractional specks carried between frames
        self._slump_credit = 0.0   # fractional avalanche steps carried over
        self._last_render = None
        self._primed = False

    # ── battery ──────────────────────────────────────────────────────────────

    def _load_peaks(self):
        """Recover learned full scale from disk. Absent or corrupt is fine:
        the peaks just relearn from the floor."""
        try:
            with open(self.peak_file, 'r') as f:
                stored = json.load(f)
        except (OSError, ValueError):
            return
        for key in self.peaks:
            try:
                value = float(stored.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= self.FEASIBLE_WATTS:
                self.peaks[key] = value

    def _save_peaks(self):
        """Persist the peaks, rate-limited, written whole then renamed.

        A torn file would be read back as corrupt and silently reset the scale,
        so it is never written in place.
        """
        now = time.monotonic()
        if (self._peaks_written is not None
                and now - self._peaks_written < self.PEAK_WRITE_INTERVAL):
            return
        self._peaks_written = now
        tmp = f'{self.peak_file}.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump({k: round(v, 2) for k, v in self.peaks.items()}, f)
            os.replace(tmp, self.peak_file)
        except OSError as e:
            logging.error(f"Could not save power peaks: {e}")

    def _observe(self, status, watts):
        """Ratchet the learned peak, ignoring spikes and impossible readings.

        A candidate has to hold above the stored peak for PEAK_CONFIRM samples
        before it counts, and it is adopted at the *lowest* value across that
        run rather than the highest, so one bad sample in an otherwise sane
        stretch can only ever understate the peak.
        """
        key = {'Charging': 'charge', 'Discharging': 'discharge'}.get(status)
        if key is None or not 0.0 < watts <= self.FEASIBLE_WATTS:
            self._streak = 0          # implausible, or nothing is flowing
            return
        if watts <= self.peaks[key]:
            self._streak = 0          # nothing new to learn
            return
        if key != self._streak_key or self._streak == 0:
            # Starting a fresh run: the floor must start from this sample, not
            # carry over from the last run, or every later peak would be pinned
            # to the first one ever learned.
            self._streak_key, self._streak_floor = key, watts
        self._streak += 1
        self._streak_floor = min(self._streak_floor, watts)
        if self._streak >= self.PEAK_CONFIRM:
            previous = self.peaks[key]
            self.peaks[key] = self._streak_floor
            self._streak = 0
            logging.info(f"Power peak ({key}) {previous:.1f} -> "
                         f"{self.peaks[key]:.1f} W")
            self._save_peaks()

    def _sample(self):
        """Take up the reader's latest, if it is one this has not had yet."""
        reading = self._battery.reading
        if reading[0] == self._sampled_at:
            return
        self._sampled_at, self._status, self._capacity, self._watts = reading
        self._observe(self._status, self._watts)

    def _reference(self, status):
        """Watts that count as full scale for this direction."""
        key = {'Charging': 'charge', 'Discharging': 'discharge'}.get(status)
        learned = self.peaks.get(key, 0.0) if key else 0.0
        return max(learned, self.floor_watts)

    def _target(self):
        """Number of specks that should be settled, 0..54."""
        if self._capacity is None:
            return sum(self.heights)  # unreadable: hold the current pile
        return int(round(max(0, min(100, self._capacity)) / 100.0 * self.total))

    def _speck_life(self, status):
        """Roughly how long a speck stays on screen, in seconds.

        Converts a wanted on-screen population into a spawn rate. Time on
        screen depends on how deep the pile is, because a full pile is a short
        fall, so without this the same wattage would look about 3x busier at
        10% charge than at 100%.
        """
        mean_height = sum(self.heights) / float(self.lanes)
        reach = (self.lanes / 2.0 + 0.5)   # middle lanes to a side edge
        if status == 'Discharging':
            # Fluttering specks leave only by drifting out past the outer edge;
            # sway is purely lateral, so it does not shorten their stay.
            drift = (self.depth + 1 - mean_height) / self.float_speed
            # A bubble does the climb through the pile before any of that, and
            # at a high fill that leg is most of the speck's life. Leaving it
            # out would spawn for the short journey and get the long one, and
            # the pile would fill with far more bubbles than the wattage means.
            climb = mean_height / self.fizz_speed
            return max(drift + self._fizz_chance() * climb, 0.05)
        # Charging: fall to the surface, then roll off the side as surplus.
        fall = (self.depth + 0.5 - mean_height) / self.fall_speed
        return max(fall + reach / self.roll_speed, 0.05)

    def _stream_rate(self, status):
        """Specks/sec, chosen so the number visible at once is proportional to
        the power flowing, against a full scale this machine taught us.

        Straight proportion through the origin, which is all this needs: power
        already spans about three orders of magnitude between a fraction of a
        watt and a fully loaded CPU plus dGPU, so a linear map separates idle
        at minimum brightness from full load by more than 20x on its own. Zero
        draws nothing at all, and a tiny trickle draws one speck every few tens
        of seconds. A compressive curve like sqrt would wreck both ends - it
        turns a trickle into a stream and flattens idle-vs-full to about 5x.

        Density is a share of the *free* cells rather than an absolute count,
        because the flow has to live in whatever the pile is not using. At full
        scale that fills most of the empty space with moving specks - snow -
        while an absolute count would either saturate at high charge or look
        thin at low charge.
        """
        budget = self._speck_budget(status)
        if budget <= 0:
            return 0.0      # nowhere to draw a speck
        # At or beyond the learned peak it is full snow; it cannot get denser.
        share = min(self._watts / self._reference(status), 1.0)
        return budget * share / self._speck_life(status)

    def _speck_budget(self, status):
        """How many specks may be on screen at once at full scale.

        Falling and drifting specks need the empty space above the pile, but a
        bubble needs the pile itself, so a discharge is budgeted against a mix
        of the two in the same proportion it splits them. Without this a full
        battery had a `free` of zero and drew nothing at all - which is the one
        moment the panel most needs to say whether it is charging or draining.

        The two get their own densities because they cost opposite things. A
        speck in free space only adds to what is lit, so the empty half can be
        packed with them; a bubble unlights sand, and at the same density it
        would eat most of the pile and leave the gauge unable to say it was
        full. Sparse is enough - a couple of holes climbing each lane reads as
        fizzing without the bar ever stopping looking like a bar.
        """
        free = self.total - sum(self.heights)
        if status != 'Discharging':
            return self.max_density * free
        fizz = self._fizz_chance()
        return (fizz * self.fizz_density * sum(self.heights)
                + (1.0 - fizz) * self.max_density * free)

    # ── pile mechanics ───────────────────────────────────────────────────────
    @property
    def total(self):
        return self.lanes * self.depth

    def _pick_lane(self, eligible):
        """Weighted lane choice, biased to the middle, limited to `eligible`."""
        weights = [self.lane_weights[i] if i in eligible else 0.0
                   for i in range(self.lanes)]
        total = sum(weights)
        if total <= 0.0:
            return None
        r = random.random() * total
        acc = 0.0
        for lane, weight in enumerate(weights):
            acc += weight
            if r < acc:
                return lane
        return max(eligible)  # float rounding fell off the end

    def _roll_downhill(self, lane):
        """Follow the slope to a local low point.

        A grain landing on a slope does not stay put: it slides to whichever
        neighbour is lower, the same way a falling-sand grain goes diagonally
        when the cell straight below is taken. Heights strictly decrease along
        the walk, so it always terminates.
        """
        while True:
            lower = [i for i in (lane - 1, lane + 1)
                     if 0 <= i < self.lanes and self.heights[i] < self.heights[lane]]
            if not lower:
                return lane
            lane = min(lower, key=lambda i: self.heights[i])

    def _settle(self, lane):
        """Rest a speck on the pile, rolling downhill first.

        A speck that would come to rest past the outer edge has settled
        offscreen, so it gets pushed into whatever space is left rather than
        being lost. That is what guarantees 100% lights all 54 cells even
        though the middle lanes take most of the drops.
        """
        lane = self._roll_downhill(lane)
        if self.heights[lane] >= self.depth:
            spaces = [i for i in range(self.lanes) if self.heights[i] < self.depth]
            if not spaces:
                return False
            # Emptiest lane wins, ties break toward where the speck landed.
            lane = min(spaces, key=lambda i: (self.heights[i], abs(i - lane)))
        self.heights[lane] += 1
        return True

    def _unstable_edges(self):
        """(drop, from, to) for every slope steeper than the angle of repose."""
        edges = []
        for lane in range(self.lanes):
            for other in (lane - 1, lane + 1):
                if 0 <= other < self.lanes:
                    drop = self.heights[lane] - self.heights[other]
                    if drop > self.repose:
                        edges.append((drop, lane, other))
        return edges

    def _topple_once(self):
        """Shed one grain off the steepest overhanging edge."""
        edges = self._unstable_edges()
        if not edges:
            return False
        steepest = max(edge[0] for edge in edges)
        # Ties pick at random so cascades don't always run the same way.
        _, lane, other = random.choice([e for e in edges if e[0] == steepest])
        self.heights[lane] -= 1
        self.heights[other] += 1
        return True

    def _avalanche(self, dt):
        """Let steep edges collapse into shorter neighbours, a grain at a time.

        This is what makes the pile behave like sand rather than a bar chart:
        the drop weights decide where grains arrive, this decides where they
        can stay. Rate-limited so a slump is visible as it runs, rather than
        snapping to a stable profile within one frame.
        """
        self._slump_credit += self.slump_rate * dt
        while self._slump_credit >= 1.0:
            if not self._topple_once():
                # Already stable - don't bank credit, or a long quiet spell
                # would buy an instant avalanche the moment one appears.
                self._slump_credit = 0.0
                return
            self._slump_credit -= 1.0

    def _settle_fully(self):
        """Relax to a stable profile immediately, for use before the first frame."""
        for _ in range(self.total * self.lanes):
            if not self._topple_once():
                return

    def _side_dir(self, lane):
        """Which way a speck leaves when the surface gives it no preference:
        toward the nearer side edge."""
        return -1.0 if lane < (self.lanes - 1) / 2.0 else 1.0

    def _surface(self, lane):
        """Height of a lane, with the void past either edge counting as below
        the empty floor. A grain that reaches the outermost lane has nothing
        holding it in, so the edge is a cliff rather than a wall."""
        if 0 <= lane < self.lanes:
            return self.heights[lane]
        return -1

    def _roll_start(self, lane):
        """Which way a surplus grain sets off, or None if it is already at rest.

        Downhill wherever the pile itself offers one. The drop over the panel
        edge deliberately does not compete for that: it is a bottomless one, so
        letting it into the comparison would win every time and pitch a grain
        standing on a tall outer lane straight over the side, when running down
        the slope beside it is the path the eye expects. The edge is where a
        grain goes when the pile has nothing lower left to offer, not the first
        thing it reaches for.

        Level ground still carries it - a grain arriving with the momentum of
        its own fall does not stop dead on the flat - and it keeps to the
        nearer side, which is what walks the stream off a flat pile instead of
        parking it where it landed. Landing in a dip with both neighbours
        higher means it has nowhere to go at all.
        """
        here = self.heights[lane]
        inside = [d for d in (-1.0, 1.0) if 0 <= lane + int(d) < self.lanes]
        lower = [d for d in inside if self.heights[lane + int(d)] < here]
        if lower:
            return min(lower, key=lambda d: self.heights[lane + int(d)])
        # Nothing in the pile is lower. Level lanes still carry it, and at an
        # outer lane the void alongside always qualifies - that is the way out.
        level = [d for d in (-1.0, 1.0) if self._surface(lane + int(d)) <= here]
        if not level:
            return None
        side = self._side_dir(lane)
        return side if side in level else level[0]

    def _rolls_on(self, speck):
        """Whether a grain can keep going, having just entered a new lane.

        Downhill and across the flat it carries on; anything higher stops it,
        and the foot of that rise is where it comes to rest. It can never be
        sent back the way it came - the lane behind it is by definition not
        lower - so this needs no turn case and cannot oscillate.
        """
        lane = speck['cell']
        return self._surface(lane + int(speck['dir'])) <= self._surface(lane)

    def _drop(self, lane, alpha=1.0):
        """Release a speck from beyond the outer edge, falling inward. Only a
        solid one (alpha 1) can settle; a translucent one is flow."""
        self.falling.append({'lane': lane, 'pos': float(self.depth) + 0.5,
                             'alpha': alpha})

    def _fill(self):
        """Share of the panel the pile occupies, 0..1."""
        return sum(self.heights) / float(self.total)

    def _fizz_chance(self):
        """Odds that a departing speck starts as a bubble rather than a floater.

        The pile is lit and the space above it is not, so the two ways out are
        legible in opposite conditions: a white speck drifting off the surface
        needs dark space to be seen against, and a black void rising through
        the sand needs sand to rise through. Tying the choice to the fill means
        each is used exactly where it reads - all floaters when the pile is a
        sliver, all bubbles when it is nearly the whole panel, and at no point
        a discharge that looks like nothing is happening.
        """
        return self._fill()

    def _float_from(self, lane, pos, alpha=1.0):
        """Put a white speck at `pos` in `lane` and let it flutter away."""
        self.floating.append({
            'alpha': alpha,
            'lane': float(lane),      # the lane it left; sway is around this
            'pos': float(pos),
            'sway': random.uniform(*self.sway_width),
            'rate': random.uniform(*self.sway_rate),
            'phase': random.uniform(0.0, 2.0 * math.pi),
            'speed': self.float_speed * random.uniform(1.0 - self.float_jitter,
                                                       1.0 + self.float_jitter),
            'age': 0.0,
        })

    def _lift(self, lane, alpha=1.0):
        """Start a speck on its way out of the pile.

        Either it peels straight off the surface, or it begins as a bubble at
        the floor and has to climb through the sand first. Both end the same
        way - a white speck fluttering off the top - so this only decides where
        the journey starts.

        A bubble picks its own lane, evenly across every lane with sand in it,
        rather than using the one the grain left from. The lane weights are
        there to shape the heap, and steering the fizz with them too puts every
        void in the middle two lanes - better than two thirds of each one dark
        at full load, with the outer lanes never fizzing at all. Nothing ties a
        rising void to the column that happened to lose a grain off its top.
        """
        deep = [i for i in range(self.lanes) if self.heights[i] > 0]
        if deep and random.random() < self._fizz_chance():
            # Always solid; a translucent one is spawned at its alpha as odds.
            if random.random() < alpha:
                self.bubbles.append({'lane': random.choice(deep), 'pos': 0.0,
                                     'alpha': 1.0})
        else:
            self._float_from(lane, self.heights[lane], alpha)

    @staticmethod
    def _sway_lane(speck):
        """Where a fluttering speck is across the lanes, right now."""
        return speck['lane'] + speck['sway'] * math.sin(
            speck['phase'] + speck['rate'] * speck['age'])

    # ── simulation ───────────────────────────────────────────────────────────
    def _advance(self, dt, target):
        """Move everything already in flight, settling or shedding on arrival."""
        landed = []
        for speck in self.falling:
            speck['pos'] -= self.fall_speed * dt
            if speck['pos'] <= self.heights[speck['lane']]:
                landed.append(speck)
        for speck in landed:
            self.falling.remove(speck)
            lane = speck['lane']
            if speck['alpha'] >= 1.0 and sum(self.heights) < target:
                self._settle(lane)
            else:
                # Surplus: this speck is only here to show current flowing, so
                # it runs down the surface until the slope stops giving.
                direction = self._roll_start(lane)
                if direction is not None:
                    self.rolling.append({'lane': float(lane), 'cell': lane,
                                         'dir': direction,
                                         'alpha': speck['alpha']})

        moving = []
        for speck in self.rolling:
            speck['lane'] += speck['dir'] * self.roll_speed * dt
            # The slope is only reconsulted on entering a new lane, so a grain
            # crosses each one in a straight line and the pile can shift under
            # it mid-roll without the direction stuttering every frame.
            cell = int(round(speck['lane']))
            if cell != speck['cell']:
                speck['cell'] = cell
                if 0 <= cell < self.lanes and not self._rolls_on(speck):
                    continue        # come to rest at the foot of a rise
            moving.append(speck)
        self.rolling = [s for s in moving
                        if -0.5 <= s['lane'] <= self.lanes - 0.5]

        # Bubbles climb through the settled sand and burst into a floater at the
        # surface. The surface is read fresh every frame rather than fixed at
        # spawn, so a pile that shrinks or slumps under a rising bubble lets it
        # out early instead of stranding it above its own sand.
        rising = []
        for bubble in self.bubbles:
            bubble['pos'] += self.fizz_speed * dt
            surface = self.heights[bubble['lane']]
            if bubble['pos'] >= surface:
                self._float_from(bubble['lane'], surface, bubble['alpha'])
            else:
                rising.append(bubble)
        self.bubbles = rising

        for speck in self.floating:
            speck['age'] += dt
            speck['pos'] += speck['speed'] * dt
        # Culled only on the way out. A speck whose sway carries it past a side
        # edge is merely not drawn for those frames - dropping it there would
        # delete specks that were about to swing back into view.
        self.floating = [s for s in self.floating if s['pos'] <= self.depth + 1]

    def _emit(self, dt, status, target):
        """Spawn this frame's specks.

        Correcting the pile toward the battery reading always takes precedence
        over the cosmetic stream, so the gauge cannot get stuck reading wrong
        no matter what the status says. The rate is in specks' worth of light:
        a solid speck spends a whole one, a translucent particle only its
        alpha, which is what makes the flow finer without making it brighter.
        """
        flowing = status in ('Charging', 'Discharging')
        rate = self._stream_rate(status) if flowing else 0.0
        if sum(self.heights) != target:
            # The pile must reach the reading even when nothing is flowing, so
            # convergence gets a floor. At rest and already correct, the rate
            # stays exactly zero and nothing is drawn at all.
            rate = max(rate, self.IDLE_RATE)
        if rate <= 0.0:
            return

        self._credit += rate * dt
        alpha = self.particle_alpha
        while True:
            settled = sum(self.heights)
            # Grains already on their way count toward the reading, or a fast
            # stream would drop far more solid grains than the pile has room
            # for, and the spare ones would pour off as solid surplus.
            arriving = sum(1 for s in self.falling if s['alpha'] >= 1.0)
            if settled + arriving < target:
                if self._credit < 1.0:
                    break
                self._credit -= 1.0
                self._drop(self._pick_lane(set(range(self.lanes))))
            elif settled > target:
                lane = self._pick_lane({i for i in range(self.lanes)
                                        if self.heights[i] > 0})
                if lane is None or self._credit < 1.0:
                    break
                self._credit -= 1.0
                self.heights[lane] -= 1
                self._lift(lane)
            elif status == 'Charging':
                # Enough on its way: keep pouring, but as translucent flow that
                # rolls off, which is what shows charge still flowing in at a
                # full-ish battery.
                if self._credit < alpha:
                    break
                self._credit -= alpha
                self._drop(self._pick_lane(set(range(self.lanes))), alpha)
            elif status == 'Discharging':
                # At target: flow keeps peeling off the surface without the
                # pile actually shrinking.
                lane = self._pick_lane({i for i in range(self.lanes)
                                        if self.heights[i] > 0})
                if lane is None or self._credit < alpha:
                    break
                self._credit -= alpha
                self._lift(lane, alpha)
            else:
                break
        # Waiting on grains in flight is not a reason to bank a burst of
        # spawns for the moment they land.
        self._credit = min(self._credit, 1.0)

    def _prime(self, target):
        """Fill the pile to the current charge without animating up from empty,
        so the gauge is correct from the first frame."""
        while sum(self.heights) < target:
            spaces = {i for i in range(self.lanes) if self.heights[i] < self.depth}
            lane = self._pick_lane(spaces)
            if lane is None or not self._settle(lane):
                break
        self._settle_fully()   # show a stable slope on the very first frame
        self._primed = True

    # ── render ───────────────────────────────────────────────────────────────
    def render(self, width):
        if self.depth is None:
            self.depth = width
        self._sample()
        status = self._status
        target = self._target()

        if not self._primed:
            self._prime(target)

        # Clamped at both ends. The upper clamp stops a stall (the artwork or
        # overview covering it, a config reload) teleporting the pile; the lower one means a clock that
        # somehow steps backwards costs one still frame, rather than driving
        # the speck credit negative and wedging the simulation for good.
        now = time.monotonic()
        dt = 0.0 if self._last_render is None else min(now - self._last_render, self.MAX_DT)
        dt = max(dt, 0.0)
        self._last_render = now

        self._advance(dt, target)
        self._emit(dt, status, target)
        # Last, so any cliff this frame's arrivals or removals just created
        # starts collapsing rather than standing until the next frame.
        self._avalanche(dt)

        grid = np.zeros((self.height, width))
        for lane in range(self.lanes):
            grid[lane, :self.heights[lane]] = 255.0

        # Straight after the pile and before any white speck: a bubble is a hole
        # in the sand, so it has to be able to darken a pixel the pile just lit,
        # and must not be able to darken one a speck is using. Whole cells,
        # fully dark, so it stands out against sand at full brightness.
        for bubble in self.bubbles:
            cell = int(bubble['pos'])
            if 0 <= cell < width:
                grid[bubble['lane'], cell] = 0.0

        for speck in self.falling:
            self._splat(grid, speck['lane'], speck['pos'], 255.0 * speck['alpha'])
        for speck in self.rolling:
            lane = int(round(speck['lane']))
            if 0 <= lane < self.lanes:
                grid[lane, min(self.heights[lane], width - 1)] += 255.0 * speck['alpha']
        for speck in self.floating:
            lane = int(round(self._sway_lane(speck)))
            if 0 <= lane < self.lanes:
                self._splat(grid, lane, speck['pos'], 255.0 * speck['alpha'])

        np.clip(grid, 0.0, 255.0, out=grid)
        return Image.fromarray(np.rint(grid).astype(np.uint8), 'L')

    @staticmethod
    def _splat(grid, lane, pos, light):
        """Add `light` at fractional depth `pos` in `lane`, shared between the
        two cells either side of it. Cell i spans [i, i+1), so its centre is
        i + 0.5 and a speck there lights that cell alone."""
        centre = pos - 0.5
        first = math.floor(centre)
        share = centre - first
        depth = grid.shape[1]
        if 0 <= first < depth:
            grid[lane, first] += light * (1.0 - share)
        if 0 <= first + 1 < depth:
            grid[lane, first + 1] += light * share

