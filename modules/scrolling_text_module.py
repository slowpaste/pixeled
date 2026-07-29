import time

from modules.module_base import ModuleBase
from PIL import Image
from utils.tiny_font import draw_tiny_text

class ScrollingTextModule(ModuleBase):
    def __init__(self, height, scroll_speed=6.0):
        super().__init__(height)
        # Scroll rate in pixels per SECOND, not per frame. The panel runs at
        # ~5.9fps in greyscale but ~50fps in black/white, so the old per-frame
        # step scrolled ~8x faster in one mode than the other. 6.0 reproduces
        # the greyscale cadence `offset += 1` used to give.
        self.scroll_speed = scroll_speed
        self.offset = 0.0
        self._last_render = None

    def render(self, width):
        image = super().render(width)
        text = "FRAMEWORK"
        text_width = len(text) * 4
        # The font draws on whole pixels, so the accumulator carries the
        # fraction and only the draw position is floored.
        x = width - int(self.offset % (text_width + width))
        draw_tiny_text(image, text, x, 0)

        # Elapsed wall time since the last frame. Clamped so a stall (startup,
        # config reload, mode switch) can't jump the text a long way.
        now = time.monotonic()
        dt = 0.0 if self._last_render is None else min(now - self._last_render, 0.25)
        self._last_render = now
        self.offset += self.scroll_speed * dt
        return image
