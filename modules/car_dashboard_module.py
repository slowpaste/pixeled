# File: modules/car_dashboard_module.py

from modules.module_base import ModuleBase
from PIL import Image
# Import the OutGaugeReader to get data
from utils.udp_outgauge_utility import OutGaugeReader

class CarDashboardModule(ModuleBase):
    """
    A module that displays key car telemetry (speed, rpm, turbo, fuel, throttle, brake, clutch)
    as horizontal bar graphs in a specified height (default=8).
    """

    def __init__(self,
                 height=8,
                 max_speed=100.0,
                 max_rpm=8000.0,
                 max_turbo=30.0):
        """
        :param height: The vertical space (rows) this module occupies. Defaults to 8.
        :param max_speed: Speed that corresponds to 100% on the bar graph.
        :param max_rpm: RPM that corresponds to 100%.
        :param max_turbo: Turbo PSI that corresponds to 100%.
        """
        super().__init__(height)

        # Initialize the outgauge reader. 
        # Typically you'd start it externally (e.g. in main.py), 
        # but for self-contained usage, we can start it here or lazily.
        self.outgauge = OutGaugeReader()
        self.outgauge.start()  # Start listening right away

        self.max_speed = max_speed
        self.max_rpm = max_rpm
        self.max_turbo = max_turbo

    def _value_to_bar(self, ratio, width):
        ratio_clamped = max(0.0, min(1.0, ratio))
        filled_len = int(ratio_clamped * width)
        return [255]*filled_len + [0]*(width - filled_len)

    def render(self, width):
        # Always fetch the latest telemetry
        data = self.outgauge.get_data()  

        speed    = data["speed"]
        rpm      = data["rpm"]
        turbo    = data["turbo"]
        fuel     = data["fuel"]
        throttle = data["throttle"]
        brake    = data["brake"]
        clutch   = data["clutch"]

        # Convert raw values into 0..1 ratio for drawing
        speed_ratio    = speed    / self.max_speed   if self.max_speed   > 0 else 0
        rpm_ratio      = rpm      / self.max_rpm     if self.max_rpm     > 0 else 0
        turbo_ratio    = turbo    / self.max_turbo   if self.max_turbo   > 0 else 0
        fuel_ratio     = fuel  # fuel is already [0..1] in OutGauge
        throttle_ratio = throttle
        brake_ratio    = brake
        clutch_ratio   = clutch

        image = Image.new('L', (width, self.height), 0)

        # We'll create up to 7 bars (rows). 
        # If height < 7, you won't see them all. 
        # If height==8, you get an extra blank row at the bottom.
        bar_rows = [
            self._value_to_bar(speed_ratio,    width),
            self._value_to_bar(rpm_ratio,      width),
            self._value_to_bar(turbo_ratio,    width),
            self._value_to_bar(fuel_ratio,     width),
            self._value_to_bar(throttle_ratio, width),
            self._value_to_bar(brake_ratio,    width),
            self._value_to_bar(clutch_ratio,   width),
        ]

        # If there's space for row 8, add a blank row
        total_rows = min(self.height, 8)
        if total_rows == 8:
            bar_rows.append([0]*width)

        # Paste bar rows into the image
        for y in range(total_rows):
            row_data = bar_rows[y]
            for x, val in enumerate(row_data):
                image.putpixel((x, y), val)

        return image
