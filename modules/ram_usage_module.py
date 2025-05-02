from modules.module_base import ModuleBase
from utils.metering_utility import MeteringUtility
from PIL import Image

class RamUsageModule(ModuleBase):
    def __init__(self, height=1):
        super().__init__(height)  # RAM usage module height
        self.metering_utility = MeteringUtility(min_value=0, max_value=100, num_pixels=9, height=height)

    def get_ram_usage(self):
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split()
            meminfo[parts[0].strip(':')] = int(parts[1])
        mem_total = meminfo['MemTotal']
        mem_free = meminfo['MemFree'] + meminfo['Buffers'] + meminfo['Cached']
        mem_used = mem_total - mem_free
        ram_usage = (mem_used / mem_total) * 100
        return ram_usage

    def render(self, width):
        ram_usage = self.get_ram_usage()
        metering_image = self.metering_utility.render(width, ram_usage)
        image = Image.new('L', (width, self.height), 0)
        for y in range(self.height):
            for x in range(width):
                image.putpixel((x, y), metering_image[y][x])
        return image
