from collections import deque

from modules.module_base import ModuleBase
from PIL import Image, ImageChops
from utils.sound_visualizer import sound_visualizer
from utils.ticker_line import TickerLine
from utils.tiny_font import sanitize_tiny_text, tiny_text_width
import math
import requests
import time
import threading
import socket

class TransitIncidentsModule(ModuleBase):
    """WMATA bus incidents, said when they are new and then left alone.

    A new report is an alert: the row flashes, and then the reports that came
    in together scroll past twice and are not repeated. The rest of the time
    there is nothing to say, and the row is the sound visualizer, which
    scrolls in once the last words have gone and rests there - showing the
    volume in silence - until something new pushes it off. "No incidents reported" and
    the fetch's own troubles are not news, and are not shown.

    The row is a TickerLine, shared with the overlays: the overview and the
    artwork take it at a word boundary and give it back, and a report cut off
    by them resumes at the word it was cut before.
    """

    PASSES = 2              # times each announcement scrolls past
    FLASHES = 3
    FLASH_PERIOD = 0.4      # seconds a flash takes, lit and then dark
    FLASH_LIT = 0.35        # fraction of it at full brightness
    FLASH_FADE = 0.3        # fraction of it fading out after that

    def __init__(self, api_key, scroll_speed=24.0):
        self.api_key = api_key
        self.height = 5
        # Pixels per SECOND, so the rate holds whatever the panel's frame rate.
        # A judgement call about real LEDs, not about the maths: text that
        # reads fine simulated on a monitor smears on the panel, because the
        # LEDs and the eye both hold a lit pixel longer than a browser does. It
        # is settable at runtime by the pixeled-speed script for exactly that
        # reason - the readable ceiling has to be found on the hardware.
        self._line = TickerLine(9, float(scroll_speed))
        self._visualizer = sound_visualizer()
        self.incidents = []
        self._seen = set()          # descriptions in the last good fetch
        self._seen_lock = threading.Lock()
        self._news = deque()        # batches of new reports, from the fetchers
        self._queue = []            # texts still to say in this announcement
        self._text = None           # the text on the line, or None
        self._skip = 0              # characters of it already said
        self._text_at = 0           # u it starts at
        self._vis_at = None         # u of the visualizer, while it has the line
        self._flash = None          # seconds into the alert, while flashing
        self._clock = None
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
        self._note_new(self.incidents)

    def _note_new(self, incidents):
        """Queue the reports not in the last good fetch, as one announcement.

        Everything at startup is new. A report that drops out of the feed and
        comes back later is new again, which is what it would be to a reader.
        """
        with self._seen_lock:
            new = [text for text in incidents if text not in self._seen]
            self._seen = set(incidents)
        new = [text for text in map(sanitize_tiny_text, new) if text]
        if new:
            self._news.append(new)

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
        print("All retry attempts failed.")

    def check_connectivity_and_fetch(self):
        if self.is_online():
            self.fetch_incidents_with_retries()
        else:
            print("Initial check: No internet connection.")
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

    def set_scroll_speed(self, px_per_second):
        """Retune the per-second scroll rate while running."""
        self._line.text_speed = float(px_per_second)

    # ── The line ─────────────────────────────────────────────────────────────

    def _say_next(self, start):
        """Put the next text of the announcement on the line at u `start`, or
        with nothing left, have the visualizer come in and rest."""
        if self._queue:
            self._text, self._skip, self._text_at = self._queue.pop(0), 0, start
            self._vis_at = None
        else:
            self._text = None
            self._vis_at = self._line.rest(self._visualizer)

    def hand_over(self, width):
        """Give up the line at the next word, for an overlay's title.

        Returns the words still on the panel, the x the next one would have
        been drawn at, which is where the title starts, and the speed they are
        moving at. The report picks up from that word when the line comes back.
        """
        self._line.width = width
        handback, cut = self._line.give(0.0)
        drawn = self._line.drawn
        if self._text is not None and drawn and drawn[-1][0] is not self._visualizer:
            if cut is None:
                # All of it had come on: what is left to say comes after.
                self._text = self._queue.pop(0) if self._queue else None
                self._skip = 0
            elif cut[0] == len(drawn) - 1:
                self._skip += cut[1]
        self._vis_at = None
        if self._flash is not None:
            self._flash = 0.0       # the alert is seen whole, when it is seen
        return handback['pieces'], handback['x'], handback['speed']

    def take_back(self, pieces, x, width, speed, accel, dt):
        """Resume after an overlay's words, the report's next word at x.

        The words arrive moving at `speed`, which the line ramps from. `dt` is
        the overlay's last frame time, standing in for the one this module
        missed.
        """
        self._line.width = width
        start = self._line.take_back(pieces, x, speed, dt)
        if self._text is not None:
            self._text_at = start
        else:
            self._vis_at = self._line.rest(self._visualizer)
        # The panel has been the overlay's since this last drew, so its own
        # clock is stale and would lurch the text the full clamp.
        self._clock = time.monotonic() - dt

    def _flash_level(self):
        phase = (self._flash % self.FLASH_PERIOD) / self.FLASH_PERIOD
        fade = max(0.0, phase - self.FLASH_LIT) / self.FLASH_FADE
        return round(255 * max(0.0, 1.0 - fade))

    def render(self, width):
        now = time.monotonic()
        dt = 0.0 if self._clock is None else min(max(now - self._clock, 0.0), 0.25)
        self._clock = now
        line = self._line
        line.width = width
        image = super().render(width)

        if (self._flash is None and self._text is None and not self._queue
                and self._news):
            batch = []
            while self._news:
                batch += self._news.popleft()
            self._queue = batch * self.PASSES
            self._flash = 0.0
        if self._flash is not None:
            self._flash += dt
            if self._flash >= self.FLASHES * self.FLASH_PERIOD:
                self._flash = None
                self._say_next(line.follow_on())

        if self._text is not None:
            text = self._text[self._skip:]
            if line.x(self._text_at) + tiny_text_width(text) <= 0:
                # Gone past: the next pass comes straight in at the right edge.
                self._say_next(math.floor(line.scroll))
        if self._text is None and self._vis_at is None:
            self._vis_at = line.rest(self._visualizer)

        if self._text is not None:
            line.move(dt, self._text_at, line.text_speed)
            placed = [(self._text[self._skip:], self._text_at)]
        else:
            line.move(dt, self._vis_at + width, 0.0)
            placed = [(self._visualizer, self._vis_at)]
        line.draw(image, placed)

        if self._flash is not None:
            image = ImageChops.lighter(
                image, Image.new('L', image.size, self._flash_level()))
        return image
