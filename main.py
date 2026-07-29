#!/home/ecca/pixeled/venv/bin/python

import serial
from serial.tools import list_ports
import numpy as np
import time
import json
import importlib
from compositor import Compositor
from PIL import Image
import re
import inspect
import logging
import threading
import os
import shutil
from utils.udp_outgauge_utility import outgauge_reader

# Panel geometry and the input-module serial protocol (FrameworkComputer/inputmodule-rs)
WIDTH, HEIGHT = 9, 34
FWK_MAGIC = (0x32, 0xAC)
DRAW_BW = 0x06      # magic + 0x06 + 39 bytes -> whole frame, 1 bit per pixel
STAGE_COL = 0x07    # magic + 0x07 + column index + 34 greyscale bytes
FLUSH_COLS = 0x08   # magic + 0x08 + 0x00 -> display the staged columns
LED_MATRIX_VID, LED_MATRIX_PID = 0x32AC, 0x0020

# The panel accepts ~59 commands/sec regardless of payload size, so frame rate
# is set purely by commands per frame. A greyscale frame costs 10 (nine staged
# columns plus a flush) because the firmware zeroes its staging buffer on every
# flush; a black/white frame costs 1. Hence ~5.9fps vs ~50fps.
MODE_GREY, MODE_BW = 'grey', 'bw'
MODE_FPS = {MODE_GREY: 6.0, MODE_BW: 50.0}
MODE_POLL_INTERVAL = 1.0  # seconds between checks of the mode file

BEAMNG_POLL_INTERVAL = 10  # seconds

# Ordered 4x4 Bayer thresholds, tiled to the panel. Needed because a flat
# threshold erases the noise band entirely - it sits well below mid-grey.
_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
BAYER_THRESHOLD = np.tile(_BAYER4, (HEIGHT // 4 + 1, WIDTH // 4 + 1))[:HEIGHT, :WIDTH] * (255.0 / 16.0)

# Define global variables at the module level
config_changed = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODE_FILE = os.path.join(BASE_DIR, 'display_mode')

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

def swap_config(use_beamng_config):
    """Swap between normal and BeamNG configs"""
    try:
        if use_beamng_config:
            if os.path.exists('config.json') and not os.path.exists('config.json.normal'):
                shutil.copy2('config.json', 'config.json.normal')
            shutil.copy2('config.json.beamng', 'config.json')
            logging.info("Switched to BeamNG configuration")
            return True
        else:
            if os.path.exists('config.json.normal'):
                shutil.copy2('config.json.normal', 'config.json')
                logging.info("Reverted to normal configuration")
                return True
    except Exception as e:
        logging.error(f"Error swapping config: {e}")
    return False

def start_beamng_monitor():
    def monitor_thread():
        global config_changed  # Declare global at the beginning of the function
        was_beamng_running = False
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
            logging.info(f"Connected to LED matrix on {device}")
        except (serial.SerialException, OSError) as e:
            self._retry_at = now + self.RECONNECT_DELAY
            logging.error(f"Could not open LED matrix: {e}")
            return None
        return self._ser

    def _drop(self, err):
        logging.error(f"LED matrix write failed, will reconnect: {err}")
        self.close()
        self._retry_at = time.monotonic() + self.RECONNECT_DELAY

    @staticmethod
    def _pack_bw(image):
        """Dither to 1 bit and pack into the 39-byte DrawBW payload.

        Bit index is i = x + 9*y, stored at bit (i % 8) of byte (i // 8). A
        row-major flatten of the 34x9 array yields exactly that ordering, so
        packbits with little bit order does the whole thing in one pass.
        """
        bits = np.asarray(image, dtype=np.float64) > BAYER_THRESHOLD
        return np.packbits(bits.reshape(-1), bitorder='little').tobytes()

    def draw(self, image, mode):
        """Send one frame in the requested mode.

        Each command must be its own write(). Concatenating them into a single
        buffer leaves the panel frozen - the firmware parses one command per
        read from the USB endpoint and discards the rest of the buffer.

        Greyscale must restage all nine columns every frame: the firmware zeroes
        its staging buffer on each flush, so any column left out goes black.
        """
        if mode == MODE_BW:
            payload = self._pack_bw(image)
        else:
            payload = image.transpose(Image.TRANSPOSE).tobytes()
        # Mode is part of the key so a switch always forces a full redraw.
        key = (mode, payload)
        if key == self._last_frame:
            return  # nothing changed; skip the transfer entirely
        ser = self._connect()
        if ser is None:
            return
        try:
            if mode == MODE_BW:
                ser.write(bytes((*FWK_MAGIC, DRAW_BW)) + payload)
            else:
                for x in range(WIDTH):
                    ser.write(bytes((*FWK_MAGIC, STAGE_COL, x)) + payload[x * HEIGHT:(x + 1) * HEIGHT])
                ser.write(bytes((*FWK_MAGIC, FLUSH_COLS, 0x00)))
            self._last_frame = key
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

def read_mode(fallback=MODE_GREY):
    """Read the display mode written by the pixeled-mode script.

    Anything unreadable or unrecognised keeps the current mode, so a truncated
    or half-written file can't blank the panel.
    """
    try:
        with open(MODE_FILE, 'r') as f:
            value = f.read().strip().lower()
    except OSError:
        return fallback
    return value if value in MODE_FPS else fallback

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def load_modules(config, width, height):
    compositor = Compositor(width, height, config)

    # Add modules based on the configuration
    for mod in config:
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

        # Add the module to the compositor
        compositor.add_module(module_instance, position)

    # Initialize layout once
    compositor.initialize_layout()
    return compositor

def main():
    global config_changed  # Declare global at the beginning of the function

    width, height = WIDTH, HEIGHT
    config_file = 'config.json'

    # Start the BeamNG monitor thread
    monitor_thread = start_beamng_monitor()

    # Initial configuration load
    config = load_config(config_file)
    compositor = load_modules(config, width, height)
    panel = Panel()

    # Start OutGauge reader if BeamNG is running at startup
    if is_beamng_running():
        if not outgauge_reader._thread or not outgauge_reader._thread.is_alive():
            outgauge_reader.start()

    mode = read_mode()
    period = 1.0 / MODE_FPS[mode]
    logging.info(f"Display mode: {mode} ({MODE_FPS[mode]:.0f}fps target)")
    next_frame = time.monotonic()
    next_mode_check = next_frame

    try:
        while True:
            # Pick up mode changes written by the pixeled-mode script
            if time.monotonic() >= next_mode_check:
                next_mode_check = time.monotonic() + MODE_POLL_INTERVAL
                new_mode = read_mode(mode)
                if new_mode != mode:
                    mode = new_mode
                    period = 1.0 / MODE_FPS[mode]
                    next_frame = time.monotonic()
                    logging.info(f"Display mode -> {mode} ({MODE_FPS[mode]:.0f}fps target)")

            # Check if configuration has changed
            if config_changed:
                logging.info("Reloading configuration...")
                try:
                    config = load_config(config_file)
                    compositor = load_modules(config, width, height)
                    config_changed = False
                    logging.info("Configuration reloaded successfully")
                except Exception as e:
                    logging.error(f"Failed to reload configuration: {e}")

            # Render and display
            final_image = compositor.render()
            if final_image is not None:
                panel.draw(final_image, mode)

            # Pace against a fixed deadline so frame spacing stays constant.
            # A plain sleep(period) would add the render time to every frame,
            # making the cadence wobble with load.
            next_frame += period
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
