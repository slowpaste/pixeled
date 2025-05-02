from PIL import Image

class ModuleBase:
    def __init__(self, height):
        self.height = height

    def render(self, width):
        return Image.new('L', (width, self.height), 0)
