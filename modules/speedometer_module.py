from modules.module_base import ModuleBase
from PIL import Image, ImageDraw
import math
from utils.udp_outgauge_utility import get_telemetry

class SpeedometerModule(ModuleBase):
    """
    Displays a semi-circular speedometer with a rotating needle 
    representing the current speed. It mirrors the TachometerModule 
    design, but uses the 'speed' field from the OutGauge telemetry.
    """

    def __init__(self, height=9, max_speed=50.0):
        """
        :param height: The vertical space (rows) this module occupies. 
                       15 is a nice default for an arc shape.
        :param max_speed: The speed (units up to you) considered 100% on the dial.
                          e.g., 100 for mph, 50 for m/s, etc.
        """
        super().__init__(height)
        self.max_speed = max_speed

    def render(self, width):
        """
        Renders a semi-circular speedometer with an anti-aliased needle.
        """
        # 1) Create a larger canvas for supersampling (anti-aliasing).
        scale = 3
        big_width = width * scale
        big_height = self.height * scale

        big_image = Image.new('L', (big_width, big_height), 0)
        draw = ImageDraw.Draw(big_image, 'L')

        # 2) Define the arc (135° to 405°, a 270° sweep)
        pad = 2 * scale
        box = [pad, pad, big_width - pad, big_height - pad]

        arc_color = 10  # mid-gray
        start_angle = 135
        end_angle = 405  # 270 degrees total

        # Draw the faint arc
        draw.arc(box, start_angle, end_angle, fill=arc_color, width=2*scale)

        # 3) Retrieve the current speed from telemetry
        telemetry = get_telemetry()
        current_speed = telemetry.get("speed", 0.0)  # Typically in m/s or mph
        ratio = 0.0
        if self.max_speed > 0:
            ratio = min(max(current_speed / self.max_speed, 0.0), 1.0)

        # Calculate the needle angle
        needle_angle = start_angle + (end_angle - start_angle) * ratio

        # 4) Draw the needle
        rad = math.radians(needle_angle)
        cx = big_width // 2
        cy = big_height // 2
        # Slightly shorter than half the module's height
        needle_length = int((self.height // 2) * scale * 0.9)

        nx = cx + int(needle_length * math.cos(rad))
        ny = cy + int(needle_length * math.sin(rad))

        needle_color = 255  # white
        draw.line((cx, cy, nx, ny), fill=needle_color, width=1*scale)

        # 5) Downsample for anti-aliasing
        small_image = big_image.resize((width, self.height), resample=Image.Resampling.LANCZOS)
        return small_image
