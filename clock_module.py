rom module_base import ModuleBase
from PIL import Image, ImageDraw
from datetime import datetime

class ClockModule(ModuleBase):
    def __init__(self):
        super().__init__(6)  # Clock module height

    def draw_tiny_text(self, image, text, x, y):
        # Define tiny font dictionary here (or import if shared)
        tiny_font = {
            '0': ["111", "101", "101", "101", "111"],
            '1': ["010", "110", "010", "010", "111"],
            # Add remaining characters...
        }
        draw = ImageDraw.Draw(image)
        for char in text:
            char_data = tiny_font.get(char, ["000"] * 5)
            for row, line in enumerate(char_data):
                for col, pixel in enumerate(line):
                    if pixel == '1':
                        draw.point((x + col, y + row), fill=255)
            x += 4  # Move to the next character position

    def render(self, width):
        image = super().render(width)
        current_time = datetime.now().strftime("%H:%M")
        self.draw_tiny_text(image, current_time[:2], 0, 0)  # HH
        self.draw_tiny_text(image, ":", 6, 0)  # :
        self.draw_tiny_text(image, current_time[3:], 0, 5)  # MM
        return image
