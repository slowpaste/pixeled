from modules.module_base import ModuleBase
from PIL import Image, ImageDraw
from utils.tiny_font import draw_tiny_text
import requests
import time
import threading
import socket

class TransitIncidentsModule(ModuleBase):
    def __init__(self, api_key, scroll_speed=6.0):
        self.api_key = api_key
        self.height = 5
        # Scroll rate in pixels per SECOND, not per frame. The panel runs at
        # ~5.9fps in greyscale but ~50fps in black/white, so the old per-frame
        # step scrolled ~8x faster in one mode than the other. 6.0 reproduces
        # the greyscale cadence `offset += 1` used to give.
        self.scroll_speed = scroll_speed
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

    def render(self, width):
        image = super().render(width)
        if self.incidents:
            text = self.incidents[self.current_incident_index].upper()  # Ensure text is uppercase
            text_width = len(text) * 4  # Calculate text width properly
            # The font draws on whole pixels, so the accumulator carries the
            # fraction and only the draw position is floored.
            x = width - int(self.offset % (text_width + width))
            draw_tiny_text(image, text, x, 0)

            # Elapsed wall time since the last frame. Clamped so a stall
            # (startup, config reload, mode switch) can't jump the text a long
            # way, which would otherwise skip whole incidents.
            now = time.monotonic()
            dt = 0.0 if self._last_render is None else min(now - self._last_render, 0.25)
            self._last_render = now
            self.offset += self.scroll_speed * dt

            # If the text has completely scrolled past, move to the next incident
            if self.offset >= (text_width + width):
                self.offset = 0.0
                self.current_incident_index = (self.current_incident_index + 1) % len(self.incidents)

        return image
