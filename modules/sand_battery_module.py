import json
import logging
import os
import random
import time

import numpy as np
from PIL import Image

from modules.module_base import ModuleBase

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PEAK_FILE = os.path.join(BASE_DIR, 'power_peaks.json')


class SandBatteryModule(ModuleBase):
    """Battery gauge drawn as a falling-sand pile. Black/white mode only.

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
    percentage stay; the excess rolls off the side edges of the pile surface.
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
                 roll_speed=9.0, float_speed=7.0, drift_speed=3.0,
                 repose=1, slump_rate=30.0,
                 floor_watts=45.0, max_density=0.9, peak_file=None):
        super().__init__(height)
        self.lanes = height
        self.depth = None          # cells per lane; set on first render
        self.heights = [0] * self.lanes
        self.falling = []          # specks dropping in from offscreen
        self.rolling = []          # surplus sliding along the surface to an edge
        self.floating = []         # specks leaving the pile on discharge

        self.fall_speed = fall_speed
        self.roll_speed = roll_speed
        self.float_speed = float_speed
        self.drift_speed = drift_speed
        # Full scale is the hardest this machine has been seen to push, tracked
        # per direction: a charge peaks well below a full-load discharge, so
        # sharing one scale would leave charging permanently unable to reach
        # full snow. floor_watts stops a history of nothing but idling from
        # rescaling a 5W trickle into a blizzard.
        self.floor_watts = floor_watts
        # Fraction of the *empty* cells carrying specks at full scale. Below 1
        # on purpose: the pile's edge has to stay findable through the snow.
        self.max_density = max_density
        self.peak_file = DEFAULT_PEAK_FILE if peak_file is None else peak_file
        self.peaks = {'charge': 0.0, 'discharge': 0.0}
        self._load_peaks()
        self._streak = 0            # consecutive samples above the stored peak
        self._streak_key = None
        self._streak_floor = 0.0
        self._peaks_written = None

        # Cached battery state. Sampled on a cadence rather than every frame:
        # reading sysfs 200x/sec costs little here (measured 0.7% of a core),
        # but confirming a peak over 3 samples only rejects spikes if those
        # samples span a meaningful stretch of time.
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
    def _read(self, name, cast=int):
        try:
            with open(f'{self.SYS}/{name}', 'r') as f:
                return cast(f.read().strip())
        except (OSError, ValueError):
            return None

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
        """Refresh cached battery state on SAMPLE_INTERVAL."""
        now = time.monotonic()
        if (self._sampled_at is not None
                and 0.0 <= now - self._sampled_at < self.SAMPLE_INTERVAL):
            return
        self._sampled_at = now
        self._status = self._read('status', str) or 'Unknown'
        self._capacity = self._read('capacity')
        self._watts = self._power()
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
                       or self.NOMINAL_MICROVOLTS)
        return amps * (micro_volts / 1e6)

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
            outward = (self.depth + 1 - mean_height) / self.float_speed
            return max(min(outward, reach / self.drift_speed), 0.05)
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
        free = self.total - sum(self.heights)
        if free <= 0:
            return 0.0      # pile is full; there is nowhere to draw a speck
        # At or beyond the learned peak it is full snow; it cannot get denser.
        share = min(self._watts / self._reference(status), 1.0)
        visible = self.max_density * free * share
        return visible / self._speck_life(status)

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
        """Which way a speck leaves: toward the nearer side edge."""
        return -1.0 if lane < (self.lanes - 1) / 2.0 else 1.0

    def _drop(self, lane):
        """Release a speck from beyond the outer edge, falling inward."""
        self.falling.append({'lane': lane, 'pos': float(self.depth) + 0.5})

    def _lift(self, lane):
        """Peel a speck off the surface of a lane and let it float away."""
        self.floating.append({'lane': float(lane),
                              'pos': float(self.heights[lane]),
                              'dir': self._side_dir(lane)})

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
            if sum(self.heights) < target:
                self._settle(lane)
            else:
                # Surplus: this speck is only here to show current flowing, so
                # it slides across the surface and off the side edge.
                self.rolling.append({'lane': float(lane), 'dir': self._side_dir(lane)})

        for speck in self.rolling:
            speck['lane'] += speck['dir'] * self.roll_speed * dt
        self.rolling = [s for s in self.rolling
                        if -0.5 <= s['lane'] <= self.lanes - 0.5]

        for speck in self.floating:
            speck['pos'] += self.float_speed * dt
            speck['lane'] += speck['dir'] * self.drift_speed * dt
        self.floating = [s for s in self.floating
                         if s['pos'] <= self.depth + 1
                         and -0.5 <= s['lane'] <= self.lanes - 0.5]

    def _emit(self, dt, status, target):
        """Spawn this frame's specks.

        Correcting the pile toward the battery reading always takes precedence
        over the cosmetic stream, so the gauge cannot get stuck reading wrong
        no matter what the status says.
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
        while self._credit >= 1.0:
            self._credit -= 1.0
            settled = sum(self.heights)
            if settled < target:
                self._drop(self._pick_lane(set(range(self.lanes))))
            elif settled > target:
                lane = self._pick_lane({i for i in range(self.lanes)
                                        if self.heights[i] > 0})
                if lane is None:
                    break
                self.heights[lane] -= 1
                self._lift(lane)
            elif status == 'Charging':
                # At target: keep pouring, but every speck now rolls off, which
                # is what shows charge still flowing in at a full-ish battery.
                self._drop(self._pick_lane(set(range(self.lanes))))
            elif status == 'Discharging':
                # At target: specks keep peeling off the surface without the
                # pile actually shrinking.
                lane = self._pick_lane({i for i in range(self.lanes)
                                        if self.heights[i] > 0})
                if lane is None:
                    break
                self._lift(lane)
            else:
                break

    def _prime(self, target):
        """Fill the pile to the current charge without animating up from empty,
        so the gauge is correct the instant the mode switches."""
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

        # Clamped at both ends. The upper clamp stops a stall (mode switch,
        # config reload) teleporting the pile; the lower one means a clock that
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

        grid = np.zeros((self.height, width), dtype=np.uint8)
        for lane in range(self.lanes):
            grid[lane, :self.heights[lane]] = 255

        # Everything is full brightness: this module only runs in 1-bit mode,
        # so a mid-grey speck would just snap on or off at the threshold. Using
        # 0 and 255 only means the pile survives the 1-bit reduction exactly.
        for speck in self.falling:
            pos = int(speck['pos'])
            if 0 <= pos < width:
                grid[speck['lane'], pos] = 255
        for speck in self.rolling:
            lane = int(round(speck['lane']))
            if 0 <= lane < self.lanes:
                grid[lane, min(self.heights[lane], width - 1)] = 255
        for speck in self.floating:
            lane = int(round(speck['lane']))
            pos = int(speck['pos'])
            if 0 <= lane < self.lanes and 0 <= pos < width:
                grid[lane, pos] = 255

        return Image.fromarray(grid, 'L')
