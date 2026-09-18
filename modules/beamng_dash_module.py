import logging
import time

from PIL import Image, ImageChops, ImageDraw

from modules.module_base import ModuleBase
from utils.tiny_font import tiny_font
from utils.udp_outgauge_utility import get_telemetry


class BeamngDashModule(ModuleBase):
    """The whole 9x34 panel as one BeamNG dashboard.

    Designed when the panel's fast mode was 1-bit, so everything here is shaped
    rather than shaded: a mark is either a solid mass or a 50% dither, and
    anything that needs to read as "changing" moves instead of fading.

    Top to bottom, with a blank row between each section:

        rows  0- 4   tachometer, a solid vertical line sliding across
        rows  6- 9   speedometer, the same line dashed so the two never merge
        rows 11-21   6-speed H pattern, dithered rails and a solid 3x3 marker
        rows 23-33   clutch, brake and throttle, three 3px bars growing upward

    At the upshift point the tachometer band inverts on and off a few times a
    second. Blinking the bar itself would be truer to the words but nearly
    invisible: one column going dark reads as the needle having moved, not as an
    alarm. Inverting the five rows around it flashes a whole block while leaving
    the needle legible against it, and stays out of the rest of the dash.

    Both needles are positioned in half pixels: an even step lights one column,
    an odd step lights two. That doubles the effective resolution of a 9-column
    sweep to 17 stops, and the width flicker between them is what makes small
    changes visible at all on a grid this coarse.

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

    def __init__(self, height=34, max_speed=55.0, redline_floor=4500.0,
                 shift_fraction=0.92, shift_release=0.05, shift_speed=30.0,
                 neutral_return=0.8, flash_hz=8.0):
        """
        :param max_speed: m/s at the right edge of the speedometer. Only a
            starting point - the scale grows if it is ever exceeded, so the
            needle never sits pinned and silent.
        :param redline_floor: rpm the tachometer assumes as full scale until it
            has watched the engine rev higher. Keeps a car that has only idled
            from scaling 900rpm across the whole panel.
        :param shift_fraction: share of the learned redline that starts the
            tachometer flashing.
        :param shift_release: extra share it has to drop back through before the
            flash stops, so a needle sitting on the threshold does not stutter
            in and out of the warning.
        :param shift_speed: pixels/sec the marker travels along the H pattern.
            30 puts a 1-2 shift at roughly a third of a second.
        :param neutral_return: seconds the lever has to sit in neutral before it
            counts as parked there rather than passing through, and slides back
            to the centre of the gate plane. Longer than any shift, shorter than
            any deliberate stop in neutral.
        :param flash_hz: full on-off cycles per second of the upshift warning.
            Capped at whatever the measured frame rate can actually resolve, so
            the same setting blinks rather than aliases on stock firmware,
            which draws at a tenth of the patched firmware's rate.
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

        self._marker = None      # [x, y] in panel coords, fractional in flight
        self._goal = None
        self._route = []
        self._gate_x = None      # gate the last engaged gear was in
        self._neutral_dwell = 0.0
        self._last_render = None

        self._car = None
        self._redline = self.redline_floor
        self._speed_scale = self.max_speed
        self._warning = False      # past the upshift point, tachometer flashing
        self._warning_since = 0.0  # so the flash always starts on its lit half
        self._dt_ema = None        # measured frame interval, for the flash cap

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
            else:
                self._marker[0] += dx / dist * budget
                self._marker[1] += dy / dist * budget
                budget = 0.0

    def _shift_up(self, tel, gear, live, now):
        """Whether it is time to upshift, and so whether the tachometer flashes.

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

    def _flash_lit(self, now):
        """Which half of the flash cycle we are in.

        The rate is capped so a cycle always spans at least four frames. Asking
        for 8Hz at stock firmware's ~6fps would otherwise sample the cycle at
        almost exactly its own frequency and come out as an erratic stutter, or
        as nothing at all, rather than as a blink.
        """
        hz = self.flash_hz
        if self._dt_ema:
            hz = min(hz, 1.0 / (4.0 * self._dt_ema))
        return ((now - self._warning_since) * hz) % 1.0 < 0.5

    # ── drawing ──────────────────────────────────────────────────────────────
    def _draw_needle(self, draw, width, y0, rows, value, dashed):
        """A vertical line whose column is the reading.

        Half-pixel steps: an even one lights a single column, an odd one lights
        the pair it falls between. The line breathing between one and two
        columns wide is the only cue a 9-wide sweep can give for a change
        smaller than a whole pixel.
        """
        value = min(max(value, 0.0), 1.0)
        step = int(round(value * (width - 1) * 2))
        columns = (step // 2,) if step % 2 == 0 else (step // 2, step // 2 + 1)
        for row in range(rows):
            if dashed and row % 2:
                continue
            for x in columns:
                draw.point((x, y0 + row), fill=255)

    def _draw_shifter(self, draw):
        """The H pattern, plus the marker sitting on it.

        The rails are dithered to every other pixel and the marker is solid.
        With no brightness to spend, that density difference is what separates
        the guide from the reading; drawing both solid gives a lit box with a
        slightly fatter spot somewhere in it.
        """
        left, _, right = self.gates
        # Gate dither is anchored on the top gear row, not on x+y parity, so all
        # three gates carry the identical pattern and every gear end and rail
        # junction lands on a lit pixel instead of in a hole.
        for gx in self.gates:
            for y in range(self.top_y, self.bot_y + 1, 2):
                draw.point((gx, y), fill=255)
        # The rail can't do the same: three gates an odd number of columns apart
        # can't all sit on a period-2 dither, so it fills the spans between them
        # and the junctions are drawn on top.
        for x in range(left + 1, right, 2):
            draw.point((x, self.rail_y), fill=255)
        for gx in self.gates:
            draw.point((gx, self.rail_y), fill=255)

        mx, my = int(round(self._marker[0])), int(round(self._marker[1]))
        draw.rectangle([mx - 1, my - 1, mx + 1, my + 1], fill=255)

    def _draw_reverse(self, draw, width):
        """Reverse gets a letter, not a marker position.

        Three gates fill the width, and every one of their six ends is a forward
        gear, so there is no free slot to park reverse in. A scaled-up R fills
        the same box instead and can't be mistaken for a gear.
        """
        glyph = tiny_font['R']
        scale = max(1, min(self.H_ROWS // len(glyph), width // len(glyph[0])))
        x0 = (width - len(glyph[0]) * scale) // 2
        y0 = self.h_y + (self.H_ROWS - len(glyph) * scale) // 2
        for row, line in enumerate(glyph):
            for col, bit in enumerate(line):
                if bit == '1':
                    draw.rectangle([x0 + col * scale, y0 + row * scale,
                                    x0 + (col + 1) * scale - 1,
                                    y0 + (row + 1) * scale - 1], fill=255)

    def _draw_pedals(self, draw, width, tel):
        """Clutch, brake and throttle, growing up from the bottom edge.

        Three bars of a third of the width each, with no gap between them: at
        nine columns a separator would cost a third of every bar. Two neighbours
        at the same height do merge into one block, which is a fair reading of
        two pedals at the same travel.
        """
        bar_w = max(1, width // len(self.PEDALS))
        bottom = self.height - 1
        for i, pedal in enumerate(self.PEDALS):
            value = min(max(tel.get(pedal, 0.0), 0.0), 1.0)
            filled = int(round(value * self.pedal_rows))
            if not filled:
                continue
            x0 = i * bar_w
            draw.rectangle([x0, bottom - filled + 1, x0 + bar_w - 1, bottom], fill=255)

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
        self._update_marker(dt, gear)

        image = Image.new('L', (width, self.height), 0)
        draw = ImageDraw.Draw(image)
        self._draw_needle(draw, width, self.tach_y, self.TACH_ROWS,
                          tel.get('rpm', 0.0) / self._redline, dashed=False)
        self._draw_needle(draw, width, self.speed_y, self.SPEED_ROWS,
                          tel.get('speed', 0.0) / self._speed_scale, dashed=True)
        if gear <= 0:
            self._draw_reverse(draw, width)
        else:
            self._draw_shifter(draw)
        self._draw_pedals(draw, width, tel)

        if self._shift_up(tel, gear, live, now) and self._flash_lit(now):
            band = (0, self.tach_y, width, self.tach_y + self.TACH_ROWS)
            image.paste(ImageChops.invert(image.crop(band)), band[:2])
        return image
