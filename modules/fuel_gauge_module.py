from modules.module_base import ModuleBase
from PIL import Image
from utils.udp_outgauge_utility import get_telemetry

class FuelGaugeModule(ModuleBase):
    """
    Displays the current fuel level as a horizontal bar.
    Occupies 'height' rows. Recommended height: 6.
    """

    def __init__(self, height=6, max_fuel=100.0):
        """
        :param height: Vertical space used by this module. Defaults to 6.
        :param max_fuel: Maximum fuel level for scaling (if not using ratio).
        """
        super().__init__(height)
        self.max_fuel = max_fuel

    def render(self, width):
        """
        Renders a horizontal fuel gauge bar.
        """
        image = Image.new('L', (width, self.height), 0)

        # Fetch current fuel from telemetry
        telemetry = get_telemetry()
        fuel_ratio = telemetry.get("fuel", 1.0)  # Assuming fuel is [0..1]

        # Calculate filled length
        filled_length = int(min(max(fuel_ratio, 0.0), 1.0) * width)

        # Draw filled portion
        for y in range(self.height):
            for x in range(filled_length):
                image.putpixel((x, y), 255)  # White

        # Optionally, add a border or label
        # For simplicity, omitted due to limited width

        return image
