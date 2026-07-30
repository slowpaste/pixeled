import random
import time

import numpy as np
from PIL import Image

from modules.module_base import ModuleBase


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

    def __init__(self, height=6, spread=1.1, flow_rate=14.0, fall_speed=11.0,
                 roll_speed=9.0, float_speed=7.0, drift_speed=3.0,
                 repose=1, slump_rate=30.0):
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
        self.flow_rate = flow_rate
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

    def _target(self):
        """Number of specks that should be settled, 0..54."""
        capacity = self._read('capacity')
        if capacity is None:
            return sum(self.heights)  # unreadable: hold the current pile
        return int(round(max(0, min(100, capacity)) / 100.0 * self.total))

    def _stream_rate(self):
        """Specks/sec to draw. Scaled by actual current so a fast charge
        visibly pours harder than a trickle."""
        amps = abs(self._read('current_now') or 0) / 1e6
        return self.flow_rate * (0.4 + min(amps, 3.0) / 1.6)

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
        if flowing:
            rate = self._stream_rate()
        elif sum(self.heights) != target:
            rate = self.IDLE_RATE   # idle, but the reading drifted from the pile
        else:
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
        status = self._read('status', str) or 'Unknown'
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
