from modules.module_base import ModuleBase
from utils.metering_utility import MeteringUtility
from PIL import Image

class BatteryModule(ModuleBase):
    def __init__(self, height=1):
        super().__init__(height)  # Battery module height
        self.metering_utility = MeteringUtility(min_value=0, max_value=100, num_pixels=9, height=height)

    def get_battery_capacity(self):
        with open('/sys/class/power_supply/BAT1/capacity', 'r') as f:
            capacity = int(f.read().strip())
            return capacity

    def render(self, width):
        capacity = self.get_battery_capacity()
        metering_image = self.metering_utility.render(width, capacity)
        image = Image.new('L', (width, self.height), 0)
        for y in range(self.height):
            for x in range(width):
                image.putpixel((x, y), metering_image[y][x])
        return image
