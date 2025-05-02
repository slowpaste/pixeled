from modules.module_base import ModuleBase
from PIL import Image

class InfoBarsModule(ModuleBase):
    """
    Displays 3 horizontal bars for speed, fuel, and turbo (or throttle, brake, etc.).
    Each bar is ~3 rows tall -> total 9 rows. (height=9 by default)
    """

    def __init__(self, height=9, speed=0.0, max_speed=100.0, 
                 fuel=1.0, turbo=0.0, max_turbo=30.0):
        super().__init__(height)
        self.speed = speed
        self.max_speed = max_speed
        self.fuel = fuel  # [0..1]
        self.turbo = turbo
        self.max_turbo = max_turbo

    def update_values(self, speed, fuel, turbo):
        self.speed = speed
        self.fuel = fuel
        self.turbo = turbo

    def render(self, width):
        image = Image.new('L', (width, self.height), 0)

        # We have 3 bars, each ~3 rows tall
        bar_height = self.height // 3  # integer division
        # Might not be exactly 3 if height=9 is changed, but close enough.

        # Compute ratios
        speed_ratio = 0.0
        if self.max_speed > 0:
            speed_ratio = min(max(self.speed/self.max_speed, 0), 1.0)

        fuel_ratio = min(max(self.fuel, 0), 1.0)  # already 0..1

        turbo_ratio = 0.0
        if self.max_turbo > 0:
            turbo_ratio = min(max(self.turbo/self.max_turbo, 0), 1.0)

        bar_data = [
            ("speed", speed_ratio),
            ("fuel",  fuel_ratio),
            ("turbo", turbo_ratio)
        ]

        for i, (name, ratio) in enumerate(bar_data):
            row_start = i * bar_height
            row_end = min(row_start + bar_height, self.height)
            filled_len = int(ratio * width)

            # Fill these rows horizontally
            for y in range(row_start, row_end):
                for x in range(filled_len):
                    image.putpixel((x, y), 255)  # white
                # remainder is left at 0

        return image
