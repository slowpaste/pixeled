#!/usr/bin/env python3

import serial
from serial.tools import list_ports
import time
import json
import importlib
from compositor import Compositor
from modules.artwork_overlay import ArtworkOverlay
from modules.overview_overlay import OverviewOverlay
from modules.plug_overlay import PlugOverlay
from PIL import Image
import re
import inspect
import logging
import threading
import os
import shutil
import signal
from utils.udp_outgauge_utility import outgauge_reader
from utils.sound_medium import SoundMedium
from utils.plug_physics import Breach, Ripple
from utils.sound_visualizer import sound_visualizer

# Panel geometry and the input-module serial protocol (FrameworkComputer/inputmodule-rs)
WIDTH, HEIGHT = 9, 34
FWK_MAGIC = (0x32, 0xAC)
BRIGHTNESS = 0x00   # magic + 0x00 + level -> every LED's PWM scaled by level/255
STAGE_COL = 0x07    # magic + 0x07 + column index + 34 greyscale bytes
FLUSH_COLS = 0x08   # magic + 0x08 + 0x00 -> display the staged columns
VERSION = 0x20      # magic + 0x20 -> 32-byte reply: major, minor<<4|patch, pre
GLOBAL_CURRENT = 0x30  # magic + 0x30 + level -> LED current; no level asks (pixeled build)
LED_MATRIX_VID, LED_MATRIX_PID = 0x32AC, 0x0020

# Everything is drawn in greyscale. A frame is nine staged columns plus a
# flush, because the firmware zeroes its staging buffer on every flush.
#
# How fast that goes depends on the firmware. Stock v0.2.0 rewrites every LED
# over I2C after every command, ~17ms, staged columns included, so a frame
# costs ten of those: ~5.9fps. The patched build in firmware/ skips the rewrite
# for a staged column, which makes a frame one rewrite plus nine near-free
# commands - measured at ~15ms, ~67fps. It reports itself by setting the
# pre-release flag in its version reply; see firmware/ledmatrix-pixeled.patch.
FPS_PATCHED = 60.0
FPS_STOCK = 6.0

SETTINGS_POLL_INTERVAL = 1.0  # seconds between checks of the scroll speed file

# How bright the panel is, 0 to 1, as the GNOME quick settings slider has it.
# In ~/.config rather than beside scroll_speed in /opt/pixeled: it is written
# from the desktop session, and it is a preference to keep, not state.
BRIGHTNESS_FILE = os.path.expanduser('~/.config/pixeled/brightness')
BRIGHTNESS_POLL_INTERVAL = 0.1  # seconds; a slider being dragged shouldn't lag
# Slider position to hardware level. LED light output is close to linear in
# both current and PWM, and the eye is not, so a linear slider would do all
# its visible dimming in the bottom fifth of its travel.
BRIGHTNESS_CURVE = 2.2
BEAMNG_POLL_INTERVAL = 10  # seconds

# Seconds a frame has to take before it is logged. Three frames' worth:
# below that the panel's own pacing absorbs it, above it the picture
# visibly stalls, and the log says which part of the frame was to blame.
SLOW_FRAME = 0.05

# Live tuning of the visualizer and the ripple the sockets send out, written by
# the pixeled-tune sliders. Only these settings, each held to a range, and put
# back to what the code says once the file is gone. `thump`, `jack_in`,
# `jack_out`, `power_in` and `power_out` are counters: each time one goes up,
# the visualizer is struck as by a kick drum, or a socket acts as if something
# had just been plugged into it or pulled out of it, to judge a setting by.
#
# The sliders rewrite the file every second while they are open, and for as
# long as it is that fresh the music view keeps the visualizer over the whole
# panel, so what is being tuned stays in view - unless the file says
# `hold_visualizer` is false, as it does while the ripple is being tuned,
# which is best seen over whatever the panel normally shows.
TUNING_FILE = os.path.expanduser('~/.cache/pixeled/tuning.json')
TUNING_POLL_INTERVAL = 0.1  # seconds; sliders being dragged shouldn't lag
TUNING_OPEN_FOR = 3.0       # seconds since the sliders last wrote, still open
# name in the file: (class, low, high). The visualizer's are named bare, the
# ripple's and the drain's after their class.
TUNABLE = {
    'BLAST_STRENGTH': (SoundMedium, 0.0, 20.0),
    'BLAST_CORE': (SoundMedium, 0.05, 5.0),
    'BLAST_MAX': (SoundMedium, 0.05, 0.95),
    'BLAST_FREQ': (SoundMedium, 0.5, 12.0),
    'BLAST_SOFT_FREQ': (SoundMedium, 0.5, 12.0),
    'BLAST_HOLD': (SoundMedium, 0.0, 5.0),
    'BLAST_DAMPING': (SoundMedium, 0.05, 1.5),
    'SUN': (SoundMedium, 0.0, 2.0),
    'SUN_DEPTH': (SoundMedium, 0.0, 200.0),
    'SUN_FOCUS': (SoundMedium, 0.5, 4.0),
    'SUN_SHARPNESS': (SoundMedium, 0.0, 1.0),
    'SUN_TAU': (SoundMedium, 0.02, 2.0),
    'REFRACTION': (SoundMedium, 0.0, 60.0),
    # The surface is stepped explicitly: past about 85 cells/s it blows up.
    'WAVE_SPEED': (SoundMedium, 5.0, 70.0),
    'WAVE_DAMPING': (SoundMedium, 0.0, 10.0),
    # The ripple's surface is stepped twice as often, so holds up to 160.
    'Ripple.SPEED': (Ripple, 10.0, 160.0),
    'Ripple.DAMPING': (Ripple, 0.0, 6.0),
    'Ripple.SPRING': (Ripple, 0.0, 40.0),
    'Ripple.FREQ': (Ripple, 0.5, 12.0),
    'Ripple.CYCLES': (Ripple, 0.5, 8.0),
    'Ripple.ATTACK': (Ripple, 0.005, 0.5),
    'Ripple.DECAY': (Ripple, 0.02, 3.0),
    'Ripple.SOURCE': (Ripple, 1.0, 12.0),
    'Ripple.DRIVE': (Ripple, 0.0, 12000.0),
    'Ripple.PUSH': (Ripple, 0.0, 12.0),
    'Ripple.PUSH_MAX': (Ripple, 0.5, 15.0),
    'Ripple.GRID_PULL': (Ripple, 0.0, 1.0),
    'Breach.SPEED': (Breach, 5.0, 150.0),
    'Breach.PRESSURE_RISE': (Breach, 0.0, 10.0),
    'Breach.INERTIA': (Breach, 0.005, 0.2),
    'Breach.SPIN': (Breach, 0.0, 300.0),
    'Breach.SHED': (Breach, 0.01, 0.5),
    'Breach.SPIN_FADE': (Breach, 0.05, 2.0),
    'Breach.TRAIL': (Breach, 0.0, 0.4),
}
TUNED_DEFAULTS = {name: getattr(cls, name.split('.')[-1])
                  for name, (cls, _, _) in TUNABLE.items()}

# Define global variables at the module level
config_changed = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCROLL_FILE = os.path.join(BASE_DIR, 'scroll_speed')
# Absolute, so the swap can't depend on the unit's WorkingDirectory happening
# to be the install directory.
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
BEAMNG_CONFIG = os.path.join(BASE_DIR, 'config.json.beamng')
NORMAL_CONFIG = os.path.join(BASE_DIR, 'config.json.normal')
SCROLL_MIN, SCROLL_MAX = 1.0, 120.0

logging.basicConfig(filename=os.path.join(BASE_DIR, 'service.log'), level=logging.INFO,
                   format='%(asctime)s %(message)s')

def is_beamng_running():
    """Check if BeamNG.drive is currently running.

    Reads /proc/<pid>/comm directly rather than using psutil.process_iter(),
    which costs ~17ms of GIL-held CPU per call here - enough to stall a frame
    every time the monitor thread polls.
    """
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm", "rb") as f:
                if b"BeamNG" in f.read():
                    return True
        except OSError:
            pass  # process exited mid-scan, or not ours to read
    return False

def same_contents(a, b):
    """Whether two files hold identical bytes. Missing or unreadable is False."""
    try:
        with open(a, 'rb') as fa, open(b, 'rb') as fb:
            return fa.read() == fb.read()
    except OSError:
        return False

def swap_config(use_beamng_config):
    """Swap between normal and BeamNG configs"""
    try:
        if use_beamng_config:
            # The backup is refreshed on every detection, not written once and
            # kept forever. It used to be guarded on the file not existing,
            # which froze it as a snapshot of whatever the layout was the first
            # time BeamNG ever ran: every later quit restored that, silently
            # reverting any module added to config.json since.
            #
            # The guard is on contents instead. If config.json is already the
            # BeamNG layout - the service restarted mid-game, so the monitor
            # sees a fresh detection - backing it up would overwrite the real
            # layout with the dashboard and lose it for good.
            if os.path.exists(CONFIG_FILE) and not same_contents(CONFIG_FILE, BEAMNG_CONFIG):
                shutil.copy2(CONFIG_FILE, NORMAL_CONFIG)
            shutil.copy2(BEAMNG_CONFIG, CONFIG_FILE)
            logging.info("Switched to BeamNG configuration")
            return True
        else:
            if os.path.exists(NORMAL_CONFIG):
                shutil.copy2(NORMAL_CONFIG, CONFIG_FILE)
                logging.info("Reverted to normal configuration")
                return True
    except Exception as e:
        logging.error(f"Error swapping config: {e}")
    return False

def start_beamng_monitor():
    def monitor_thread():
        global config_changed  # Declare global at the beginning of the function
        # Seeded from what config.json actually holds, not assumed False. The
        # swaps are edge-triggered, so a process that starts believing BeamNG
        # was never running will not revert a config.json left on the dashboard
        # layout - which is what happens whenever the service restarts after
        # BeamNG quit. Seeding it this way turns that stale config into a
        # running -> not-running edge on the first poll, and it heals itself.
        was_beamng_running = same_contents(CONFIG_FILE, BEAMNG_CONFIG)
        while True:
            try:
                beamng_running = is_beamng_running()

                # State change detection
                if beamng_running and not was_beamng_running:
                    logging.info("BeamNG detected! Switching configuration...")
                    if swap_config(True):
                        # Signal main thread to reload
                        config_changed = True
                        # Ensure OutGauge reader is running
                        if not outgauge_reader._thread or not outgauge_reader._thread.is_alive():
                            outgauge_reader.start()

                elif not beamng_running and was_beamng_running:
                    logging.info("BeamNG closed. Reverting configuration...")
                    if swap_config(False):
                        # Signal main thread to reload
                        config_changed = True

                was_beamng_running = beamng_running
            except Exception as e:
                logging.error(f"Error in BeamNG monitor: {e}")

            time.sleep(BEAMNG_POLL_INTERVAL)

    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    return monitor

# Other functions from your existing main.py
def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def find_led_matrix():
    """Locate the LED matrix by USB VID:PID, rather than assuming /dev/ttyACM0.

    The keyboard module (32ac:0012) sits on the same bus, so ttyACM numbering
    is not stable across replugs.
    """
    for port in list_ports.comports():
        if port.vid == LED_MATRIX_VID and port.pid == LED_MATRIX_PID:
            return port.device
    return None

class Panel:
    """Persistent connection to the LED matrix.

    The port is opened once and reused. Opening a CDC-ACM device costs ~20ms,
    and the previous code did that ten times per frame (nine columns plus the
    flush), which capped the panel at ~4.6fps regardless of the loop's sleep.
    """
    RECONNECT_DELAY = 2.0  # don't retry a missing device at frame rate

    def __init__(self):
        self._ser = None
        self._last_frame = None
        self._retry_at = 0.0
        # Until a panel has answered, pace as if it were the patched firmware:
        # nothing is being sent, so there is nothing to overrun.
        self.fps = FPS_PATCHED
        self.brightness = 1.0           # 0-1, as the slider has it
        self._current_control = False   # whether the firmware can set LED current
        self._sent_brightness = None    # hardware level the panel was last given

    def _connect(self):
        if self._ser is not None:
            return self._ser
        now = time.monotonic()
        if now < self._retry_at:
            return None
        device = find_led_matrix()
        if device is None:
            self._retry_at = now + self.RECONNECT_DELAY
            return None
        try:
            self._ser = serial.Serial(device, 115200, timeout=1)
            self.fps = self._firmware_fps(self._ser)
            self._current_control = (self.fps == FPS_PATCHED
                                     and self._has_current_control(self._ser))
            logging.info(f"Connected to LED matrix on {device}, "
                         f"pacing greyscale at {self.fps:.0f}fps, dimming by "
                         f"{'LED current' if self._current_control else 'PWM'}")
            if self._current_control:
                # PWM at full scale, so every one of its 256 steps is spent on
                # the picture; how bright that is gets set by current instead.
                self._ser.write(bytes((*FWK_MAGIC, BRIGHTNESS, 255)))
            self._sent_brightness = None
            self._send_brightness()
        except (serial.SerialException, OSError) as e:
            self.close()
            self._retry_at = now + self.RECONNECT_DELAY
            logging.error(f"Could not open LED matrix: {e}")
            return None
        return self._ser

    @staticmethod
    def _firmware_fps(ser):
        """The frame rate this panel's firmware can keep up with.

        Asked on every connect rather than once, because reflashing the module
        is exactly a disconnect and reconnect. Anything short of a clear answer
        from the patched build gets the stock rate: pacing a stock panel at
        60fps would queue ten times the commands it can take, and the picture
        would fall seconds behind.
        """
        ser.reset_input_buffer()
        ser.write(bytes((*FWK_MAGIC, VERSION)))
        reply = ser.read(32)
        if len(reply) < 3:
            logging.warning("LED matrix did not report a version; assuming stock firmware")
            return FPS_STOCK
        version = f"{reply[0]}.{reply[1] >> 4}.{reply[1] & 0xF}"
        patched = reply[2] == 1
        logging.info(f"LED matrix firmware {version}"
                     f"{' (pixeled build)' if patched else ''}")
        return FPS_PATCHED if patched else FPS_STOCK

    @staticmethod
    def _has_current_control(ser):
        """Whether this build answers the global current query.

        The first pixeled builds predate it, and firmware ignores a command it
        does not know without replying, so silence means no.
        """
        ser.reset_input_buffer()
        ser.write(bytes((*FWK_MAGIC, GLOBAL_CURRENT)))
        timeout, ser.timeout = ser.timeout, 0.3
        try:
            return len(ser.read(32)) == 32
        finally:
            ser.timeout = timeout

    def set_brightness(self, level):
        """How bright the panel should be, 0 to 1. Applied straight away if a
        panel is connected, otherwise as soon as one is."""
        self.brightness = min(max(float(level), 0.0), 1.0)
        if self._ser is not None:
            self._send_brightness()

    def _send_brightness(self):
        """Tell the panel the current brightness, by whichever means it has.

        LED current where the firmware can set it, which dims without costing
        the picture any of its PWM steps. Otherwise the stock brightness
        command, which scales PWM and so does cost them - at the old default of
        20%, all but 52. Never all the way to 0 on the slider's say-so: an
        unlit panel reads as a broken one.
        """
        level = max(1, round(255 * self.brightness ** BRIGHTNESS_CURVE))
        if level == self._sent_brightness:
            return
        command = GLOBAL_CURRENT if self._current_control else BRIGHTNESS
        try:
            self._ser.write(bytes((*FWK_MAGIC, command, level)))
            self._sent_brightness = level
        except (serial.SerialException, OSError) as e:
            self._drop(e)

    def _drop(self, err):
        logging.error(f"LED matrix write failed, will reconnect: {err}")
        self.close()
        self._retry_at = time.monotonic() + self.RECONNECT_DELAY

    def draw(self, image):
        """Send one greyscale frame.

        Each command must be its own write(). Concatenating them into a single
        buffer leaves the panel frozen - the firmware parses one command per
        read from the USB endpoint and discards the rest of the buffer.

        All nine columns are restaged every frame: the firmware zeroes its
        staging buffer on each flush, so any column left out goes black.
        """
        payload = image.transpose(Image.TRANSPOSE).tobytes()
        if payload == self._last_frame:
            return  # nothing changed; skip the transfer entirely
        ser = self._connect()
        if ser is None:
            return
        try:
            for x in range(WIDTH):
                ser.write(bytes((*FWK_MAGIC, STAGE_COL, x)) + payload[x * HEIGHT:(x + 1) * HEIGHT])
            ser.write(bytes((*FWK_MAGIC, FLUSH_COLS, 0x00)))
            self._last_frame = payload
        except (serial.SerialException, OSError) as e:
            self._drop(e)

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._last_frame = None

def read_brightness(fallback):
    """Read the brightness the GNOME extension's slider wrote, 0 to 1.

    Anything unreadable keeps the current brightness, so a missing or
    half-written file can't snap the panel to some other level.
    """
    try:
        with open(BRIGHTNESS_FILE, 'r') as f:
            value = float(f.read().strip())
    except (OSError, ValueError):
        return fallback
    return min(max(value, 0.0), 1.0) if value == value else fallback

def apply_tuning(seen, plugs):
    """Apply the tuning file if it has changed since `seen`, and return what
    to pass next time: (the file's mtime, the thump counter, the sockets'
    counters, whether to hold the visualizer over the panel), or None."""
    try:
        stamp = os.stat(TUNING_FILE).st_mtime_ns
    except OSError:
        if seen is not None:
            for name, value in TUNED_DEFAULTS.items():
                setattr(TUNABLE[name][0], name.split('.')[-1], value)
        return None
    if seen is not None and stamp == seen[0]:
        return seen
    try:
        with open(TUNING_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return seen
    if not isinstance(data, dict):
        return seen
    for name, (cls, low, high) in TUNABLE.items():
        try:
            value = float(data.get(name, TUNED_DEFAULTS[name]))
        except (TypeError, ValueError):
            continue
        if value == value:
            setattr(cls, name.split('.')[-1], min(max(value, low), high))
    thump = data.get('thump')
    # Each socket can be plugged and pulled from the sliders, to see what it
    # does. Only one asked for while running acts, not one left in the file.
    sockets = {name: (plugs.jack if 'jack' in name else plugs.power,
                      'in' if name.endswith('_in') else 'out')
               for name in ('jack_in', 'jack_out', 'power_in', 'power_out')}
    counters = {name: data.get(name) for name in sockets}
    if seen is not None:
        if thump != seen[1]:
            sound_visualizer().thump()
        for name, count in counters.items():
            # A counter the file does not carry at all is not a change: an
            # older or hand-written file should not set the panel off.
            if count is not None and count != seen[2].get(name):
                port, event = sockets[name]
                port.watch.pretend(event)
    return stamp, thump, counters, data.get('hold_visualizer', True) is not False

def tuning_open():
    """Whether the pixeled-tune sliders are open, by how recently they wrote."""
    try:
        return time.time() - os.stat(TUNING_FILE).st_mtime < TUNING_OPEN_FOR
    except OSError:
        return False

def read_scroll_speed(fallback=None):
    """Read the text scroll speed written by the pixeled-speed script.

    Returns fallback when the file is absent or unusable, so a missing or
    half-written file leaves whatever config.json asked for in place rather
    than snapping the ticker to some default.
    """
    try:
        with open(SCROLL_FILE, 'r') as f:
            value = float(f.read().strip())
    except (OSError, ValueError):
        return fallback
    if not SCROLL_MIN <= value <= SCROLL_MAX:
        return fallback
    return value

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def load_modules(config, width, height):
    compositor = Compositor(width, height, config)

    # Add modules based on the configuration
    for mod in config:
        # Layouts from when the panel had a black/white mode could restrict a
        # module to one of the two. Keep the greyscale half, which was laid out
        # to fit on its own; the other half would overlap it.
        if mod.get("modes") and "grey" not in mod["modes"]:
            continue
        module_name = mod["module"]
        module_height = mod.get("height")
        position = mod.get("position")

        # Convert module name from CamelCase to snake_case
        module_filename = camel_to_snake(module_name)

        # Dynamically import the module class
        module_class = getattr(importlib.import_module(f"modules.{module_filename}"), module_name)

        # Collect parameters for the module constructor
        constructor_args = {}
        if "height" in inspect.signature(module_class).parameters:
            constructor_args["height"] = module_height
        for param in inspect.signature(module_class).parameters:
            if param in mod:
                constructor_args[param] = mod[param]

        # Create an instance of the module with the collected parameters
        module_instance = module_class(**constructor_args)

        # Ensure the module has a height attribute
        if not hasattr(module_instance, 'height'):
            raise ValueError(f"Module {module_name} does not have a height attribute")

        compositor.add_module(module_instance, position,
                              mod.get("pull_with_overview", False))

    # Initialize layout once
    compositor.initialize_layout()
    return compositor

def render_gauges(compositor, handback=None):
    """The composed gauges, the rows an overlay pulls along as it rises over
    them, and the ticker its title shares the top row with.

    `handback` is an overlay's words for the ticker to carry on from, given
    before anything draws so the ticker picks them up this very frame.
    """
    ticker = compositor.ticker()
    if ticker is not None and handback is not None:
        ticker.take_back(**handback)
    return compositor.render(), compositor.pulled_rows(), ticker

def render_backdrop(artwork, compositor, handback=None):
    """What the panel shows beneath the overview, the rows it pulls along, and
    the ticker its title shares the top row with.

    The artwork, or the gauges. The artwork has no rows to pull - it covers
    them - and while its cover is up its track title has the top row, so the
    overview shares the row with that instead. Words handed back go to the
    track if one is playing, and otherwise to the gauges' ticker, even with
    a cover still on its way out.
    """
    if handback is not None:
        if artwork.playing():
            artwork.take_back(**handback)
        else:
            ticker = compositor.ticker()
            if ticker is not None:
                ticker.take_back(**handback)
    image = artwork.render(lambda words=None: render_gauges(compositor, words))
    if image is not None:
        return image, (), artwork.ticker()
    return render_gauges(compositor)

def render_panel(overview, artwork, compositor):
    """The frame, and which of the three drew it.

    The overview comes first and covers both others; the artwork covers the
    gauges. Each slides in over what it covers rather than cutting, so it gets
    that as a callable and renders it only on the frames the slide actually
    covers ground on.
    """
    image = overview.render(
        lambda handback=None: render_backdrop(artwork, compositor, handback))
    if image is not None:
        return image, 'overview'
    image = artwork.render(lambda words=None: render_gauges(compositor, words))
    if image is not None:
        return image, 'artwork'
    return compositor.render(), 'gauges'

def panel_layout(drawn, artwork, compositor):
    """How the frame `drawn` by render_panel is laid out, for the sockets'
    effects to bring it back into place: ('split', ticker) with a title row
    over the rest - ticker being what has the row, if it can be handed over -
    or ('whole', None)."""
    if drawn == 'overview':
        return 'split', None
    if drawn == 'artwork':
        if artwork.full_panel():
            return 'whole', None
        return 'split', artwork.ticker()
    ticker = compositor.ticker()
    return ('split', ticker) if ticker is not None else ('whole', None)

def main():
    global config_changed  # Declare global at the beginning of the function

    width, height = WIDTH, HEIGHT
    config_file = CONFIG_FILE

    # Start the BeamNG monitor thread
    monitor_thread = start_beamng_monitor()

    # Initial configuration load
    config = load_config(config_file)
    compositor = load_modules(config, width, height)
    panel = Panel()

    # Not part of the layout: it takes the whole panel when the shell says
    # there is something to show, and is inert otherwise.
    artwork = ArtworkOverlay(width, height)
    # Above both: while the overview is open it has the whole panel, and
    # slides over whatever the two below were showing.
    overview = OverviewOverlay(width, height)
    # Over everything: ripples from the headphone jack and the charger's port
    # when something is plugged into them, and the panel sucked out through
    # whichever one is pulled.
    plugs = PlugOverlay(width, height)
    # For trying them out without the headphones: kill -USR1 plugs in,
    # kill -USR2 pulls out. The pixeled-tune buttons do either socket.
    signal.signal(signal.SIGUSR1, lambda *_: plugs.jack.watch.pretend('in'))
    signal.signal(signal.SIGUSR2, lambda *_: plugs.jack.watch.pretend('out'))

    # A runtime override, if one has been set; otherwise config.json wins.
    scroll_speed = read_scroll_speed()
    if scroll_speed is not None:
        compositor.set_scroll_speed(scroll_speed)
        artwork.set_scroll_speed(scroll_speed)
        overview.set_scroll_speed(scroll_speed)
        plugs.set_scroll_speed(scroll_speed)
        logging.info(f"Scroll speed: {scroll_speed:.1f} px/s")

    # Start OutGauge reader if BeamNG is running at startup
    if is_beamng_running():
        if not outgauge_reader._thread or not outgauge_reader._thread.is_alive():
            outgauge_reader.start()

    panel.set_brightness(read_brightness(panel.brightness))

    drawn = [None]      # which of the three drew the last frame
    next_frame = time.monotonic()
    next_settings_check = next_frame
    next_brightness_check = next_frame
    next_tuning_check = next_frame
    tuning = apply_tuning(None, plugs)

    try:
        while True:
            if time.monotonic() >= next_brightness_check:
                next_brightness_check = time.monotonic() + BRIGHTNESS_POLL_INTERVAL
                panel.set_brightness(read_brightness(panel.brightness))

            if time.monotonic() >= next_tuning_check:
                next_tuning_check = time.monotonic() + TUNING_POLL_INTERVAL
                tuning = apply_tuning(tuning, plugs)
                artwork.hold_full = tuning is not None and tuning[3] and tuning_open()

            if time.monotonic() >= next_settings_check:
                next_settings_check = time.monotonic() + SETTINGS_POLL_INTERVAL
                # Pick up scroll speed changes from the pixeled-speed script
                new_speed = read_scroll_speed(scroll_speed)
                if new_speed != scroll_speed:
                    scroll_speed = new_speed
                    compositor.set_scroll_speed(scroll_speed)
                    artwork.set_scroll_speed(scroll_speed)
                    overview.set_scroll_speed(scroll_speed)
                    plugs.set_scroll_speed(scroll_speed)
                    logging.info(f"Scroll speed -> {scroll_speed:.1f} px/s")

            # Check if configuration has changed
            if config_changed:
                logging.info("Reloading configuration...")
                try:
                    config = load_config(config_file)
                    compositor = load_modules(config, width, height)
                    if scroll_speed is not None:
                        # Fresh modules start from config.json, so reapply the
                        # runtime override or a reload would silently undo it.
                        compositor.set_scroll_speed(scroll_speed)
                    config_changed = False
                    logging.info("Configuration reloaded successfully")
                except Exception as e:
                    logging.error(f"Failed to reload configuration: {e}")

            # Render and display, through whatever the sockets are doing.
            spent = [0.0]       # seconds this frame spent on the modules
            def under():
                started = time.monotonic()
                image, drawn[0] = render_panel(overview, artwork, compositor)
                spent[0] += time.monotonic() - started
                return image
            began = time.monotonic()
            final_image = plugs.render(
                under, lambda: panel_layout(drawn[0], artwork, compositor))
            drew = time.monotonic()
            panel.draw(final_image)
            done = time.monotonic()
            if done - began > SLOW_FRAME:
                # Whatever stalls the panel is worth knowing about, and which
                # part of the frame it was is most of the answer.
                logging.info(
                    f"Slow frame: {1000 * (done - began):.0f}ms - modules "
                    f"{1000 * spent[0]:.0f} (slowest {compositor.slowest()}), "
                    f"sockets {1000 * (drew - began - spent[0]):.0f}, panel "
                    f"{1000 * (done - drew):.0f}")

            # Pace against a fixed deadline so frame spacing stays constant.
            # A plain sleep(period) would add the render time to every frame,
            # making the cadence wobble with load. The rate is the panel's,
            # re-read every frame because it is only known once a panel with a
            # given firmware has answered, and changes if one is reflashed.
            next_frame += 1.0 / panel.fps
            delay = next_frame - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Fell behind (config reload, scheduler hiccup). Resync instead
                # of catching up with a burst of frames.
                next_frame = time.monotonic()
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
    finally:
        panel.close()

if __name__ == "__main__":
    main()
