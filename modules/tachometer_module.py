from modules.module_base import ModuleBase
from PIL import Image, ImageDraw
import math
from utils.udp_outgauge_utility import get_telemetry

class TachometerModule(ModuleBase):
    """
    Displays a semi-circular tachometer with a rotating needle representing RPM.
    Occupies 'height' rows, recommended ~15 for a decent arc.
    """

    def __init__(self, height=15, max_rpm=8000.0):
        """
        :param height: Vertical space used by this module. Defaults to 15.
        :param max_rpm: The RPM considered 100% on the dial.
        """
        super().__init__(height)
        self.max_rpm = max_rpm

    def render(self, width):
        """
        Renders a semi-circular tachometer with an anti-aliased needle.
        """
        # Supersampling for anti-aliasing
        scale = 3
        big_width = width * scale
        big_height = self.height * scale

        # Create a large blank image
        big_image = Image.new('L', (big_width, big_height), 0)
        draw = ImageDraw.Draw(big_image, 'L')

        # Define the arc
        pad = 2 * scale
        box = [pad, pad, big_width - pad, big_height - pad]

        # Draw the faint arc
        arc_color = 20  # Mid-gray
        start_angle = 135
        end_angle = 405  # 270 degrees sweep
        draw.arc(box, start_angle, end_angle, fill=arc_color, width=2*scale)

        # Retrieve current RPM from telemetry
        telemetry = get_telemetry()
        current_rpm = telemetry.get("rpm", 0.0)
        ratio = min(max(current_rpm / self.max_rpm, 0.0), 1.0)
        needle_angle = start_angle + (end_angle - start_angle) * ratio

        # Calculate needle position
        rad = math.radians(needle_angle)
        cx = big_width // 2
        cy = big_height // 2
        needle_length = int((self.height // 2) * scale * 0.9)  # Slightly shorter than half the height

        nx = cx + int(needle_length * math.cos(rad))
        ny = cy + int(needle_length * math.sin(rad))

        # Draw the needle
        needle_color = 255  # White
        draw.line((cx, cy, nx, ny), fill=needle_color, width=1*scale)

        # Downsample for anti-aliasing
        small_image = big_image.resize((width, self.height), resample=Image.Resampling.LANCZOS)
        return small_image
