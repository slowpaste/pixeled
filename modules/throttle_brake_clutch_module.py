from modules.module_base import ModuleBase
from PIL import Image
from utils.udp_outgauge_utility import get_telemetry

class ThrottleBrakeClutchModule(ModuleBase):
    """
    Displays throttle, brake, and clutch statuses as horizontal bars.
    Occupies 'height' rows. Recommended height: 7.
    """

    def __init__(self, height=7):
        """
        :param height: Vertical space used by this module. Defaults to 7.
        """
        super().__init__(height)

    def render(self, width):
        """
        Renders three horizontal bars for throttle, brake, and clutch.
        """
        image = Image.new('L', (width, self.height), 0)

        # Fetch current statuses from telemetry
        telemetry = get_telemetry()
        throttle = telemetry.get("throttle", 0.0)  # [0..1]
        brake = telemetry.get("brake", 0.0)        # [0..1]
        clutch = telemetry.get("clutch", 0.0)      # [0..1]

        # Define bar positions
        bars = [
            throttle,
            brake,
            clutch
        ]

        # Calculate bar height per status
        rows_per_bar = self.height // len(bars)
        for idx, status in enumerate(bars):
            filled_length = int(min(max(status, 0.0), 1.0) * width)
            for y in range(idx * rows_per_bar, (idx + 1) * rows_per_bar):
                for x in range(filled_length):
                    image.putpixel((x, y), 255)  # White

        return image
