from modules.module_base import ModuleBase
from PIL import Image
from utils.tiny_font import draw_tiny_text

class ScrollingTextModule(ModuleBase):
    def __init__(self, height):
        super().__init__(height)
        self.offset = 0

    def render(self, width):
        image = super().render(width)
        text = "FRAMEWORK"
        text_width = len(text) * 4
        x = width - (self.offset % (text_width + width))
        draw_tiny_text(image, text, x, 0)
        self.offset += 1
        return image
