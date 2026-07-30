import time

from modules.module_base import ModuleBase
from PIL import Image
from utils.tiny_font import draw_tiny_text

class ScrollingTextModule(ModuleBase):
    def __init__(self, height, scroll_speed=32.0, frame_step_modes=('grey',)):
        super().__init__(height)
        # Per-refresh in greyscale (~5.9fps, so one pixel a frame is as smooth
        # as it gets), pixels per second in black/white (~50fps, where 32 px/s
        # is 0.64 px a frame and the quantising is invisible).
        self.scroll_speed = scroll_speed
        self.frame_step_modes = tuple(frame_step_modes)
        self.mode = None
        self.offset = 0.0
        self._last_render = None

    def set_mode(self, mode):
        if mode != self.mode:
            self.mode = mode
            self._last_render = None   # first frame in the new mode moves nothing

    def _advance(self):
        """Pixels to move this frame. The clock is read either way, so leaving
        a per-refresh mode can't see a stale timestamp and jump."""
        now = time.monotonic()
        last, self._last_render = self._last_render, now
        if self.mode in self.frame_step_modes:
            return 1.0
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
