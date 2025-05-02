from modules.module_base import ModuleBase
from PIL import Image
from datetime import datetime
from utils.tiny_font import draw_tiny_text

class ClockModule(ModuleBase):
    def __init__(self):
        super().__init__(11)  # Updated clock module height to 11 pixels

    def render(self, width):
        image = super().render(width)
        current_time = datetime.now().strftime("%H%M")
        
        # Draw HH at the top
        draw_tiny_text(image, current_time[:2], 0, 0)
        
        # Draw MM below HH with a 1-pixel gap
        draw_tiny_text(image, current_time[2:], 2, 6)

        return image
