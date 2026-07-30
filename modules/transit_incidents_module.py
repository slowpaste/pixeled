from modules.module_base import ModuleBase
from PIL import Image, ImageDraw
from utils.tiny_font import draw_tiny_text
import requests
import time
import threading
import socket

class TransitIncidentsModule(ModuleBase):
    def __init__(self, api_key, scroll_speed=32.0, frame_step_modes=('grey',)):
        self.api_key = api_key
        self.height = 5
        # Two different rules, because the modes are ~8x apart in frame rate:
        #
        #   grey (~5.9fps)  one pixel per refresh, which is as smooth as 6fps
        #                   can be - anything faster has to skip pixels.
        #   bw   (~50fps)   scroll_speed pixels per SECOND. At 32 px/s that is
        #                   0.64 px per frame, so a step lands every ~31ms and
        #                   the quantising is invisible.
        self.scroll_speed = scroll_speed
        self.frame_step_modes = tuple(frame_step_modes)
        self.mode = None
        self.offset = 0.0
        self._last_render = None
        self.incidents = ["Initializing..."]
        self.current_incident_index = 0
        self.check_connectivity_and_fetch()
        self.start_periodic_fetch()

    def is_online(self):
        try:
            # Try to connect to a known server (Google's DNS)
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except OSError:
            return False

    def fetch_incidents(self):
        headers = {
            'api_key': self.api_key,
        }

        conn = requests.get('https://api.wmata.com/Incidents.svc/json/BusIncidents', headers=headers)
        conn.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        data = conn.json()
        self.incidents = [incident['Description'] for incident in data.get('BusIncidents', [])]
        if not self.incidents:
            self.incidents = ["No incidents reported."]

    def fetch_incidents_with_retries(self, retries=5, delay=10):
        for attempt in range(retries):
            if not self.is_online():
                print("No internet connection. Retrying...")
                time.sleep(delay)
                continue

            try:
                self.fetch_incidents()
                print("Successfully fetched incidents.")
                return  # Exit if successful
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
        print("All retry attempts failed. Continuing with error message.")
        self.incidents = ["Error fetching incidents. Retrying..."]

    def check_connectivity_and_fetch(self):
        if self.is_online():
            self.fetch_incidents_with_retries()
        else:
            print("Initial check: No internet connection. Starting with default message.")
            self.incidents = ["No internet connection. Waiting to retry..."]
            self.start_connectivity_check_thread()

    def start_connectivity_check_thread(self):
        def check_connectivity():
            while not self.is_online():
                print("Waiting for internet connection...")
                time.sleep(10)  # Wait before checking again
            print("Internet connection established. Fetching incidents.")
            self.fetch_incidents_with_retries()

        thread = threading.Thread(target=check_connectivity)
        thread.daemon = True  # Daemonize thread to exit when the main program exits
        thread.start()

    def start_periodic_fetch(self):
        def fetch_every_hour():
            while True:
                time.sleep(3600)  # Sleep for 1 hour
                self.fetch_incidents_with_retries()

        thread = threading.Thread(target=fetch_every_hour)
        thread.daemon = True  # Daemonize thread to exit when the main program exits
        thread.start()

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            # Drop the timestamp so the first frame in the new mode moves
            # nothing. Otherwise leaving greyscale would carry its 169ms frame
            # period into the fast mode and lurch the text ~5px on every toggle.
            self._last_render = None

    def _advance(self):
        """Pixels to move this frame.

        The clock is read on every frame regardless of which rule applies, so
        that switching out of a per-refresh mode doesn't see a stale timestamp
        and jump the text the full clamp width.
        """
        now = time.monotonic()
        last, self._last_render = self._last_render, now
        if self.mode in self.frame_step_modes:
            return 1.0   # exactly one pixel per refresh, by definition smooth
        # Clamped so a stall (startup, config reload, mode switch) can't jump
        # the text a long way, which would otherwise skip whole incidents.
        dt = 0.0 if last is None else min(now - last, 0.25)
        return self.scroll_speed * dt

    def render(self, width):
        image = super().render(width)
        if self.incidents:
            text = self.incidents[self.current_incident_index].upper()  # Ensure text is uppercase
            text_width = len(text) * 4  # Calculate text width properly
            # The font draws on whole pixels, so the accumulator carries the
            # fraction and only the draw position is floored.
            x = width - int(self.offset % (text_width + width))
            draw_tiny_text(image, text, x, 0)

            self.offset += self._advance()

            # If the text has completely scrolled past, move to the next incident
            if self.offset >= (text_width + width):
                self.offset = 0.0
                self.current_incident_index = (self.current_incident_index + 1) % len(self.incidents)

        return image
