import time

from modules.module_base import ModuleBase
from PIL import Image
from utils.tiny_font import draw_tiny_text

class ScrollingTextModule(ModuleBase):
    def __init__(self, height, scroll_speed=24.0):
        super().__init__(height)
        # Pixels per second. Runtime-settable via the pixeled-speed script,
        # because how fast text can go before it smears is a property of the
        # LEDs, not something to derive.
        self.scroll_speed = scroll_speed
        self.offset = 0.0
        self._last_render = None

    def set_scroll_speed(self, px_per_second):
        """Retune the per-second scroll rate while running."""
        self.scroll_speed = float(px_per_second)

    def _advance(self):
        """Pixels to move this frame."""
        now = time.monotonic()
        last, self._last_render = self._last_render, now
        dt = 0.0 if last is None else min(now - last, 0.25)
        return self.scroll_speed * dt

    def render(self, width):
        image = super().render(width)
        text = "FRAMEWORK"
        text_width = len(text) * 4
        # The font draws on whole pixels, so the accumulator carries the
        # fraction and only the draw position is floored.
        x = width - int(self.offset % (text_width + width))
        draw_tiny_text(image, text, x, 0)
        self.offset += self._advance()
        return image
