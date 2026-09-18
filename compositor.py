import time

from PIL import Image

class Compositor:
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.modules = []
        self.config = config
        self.layout = None  # Store the layout after the initial placement
        self.pulled = set()  # modules the overview carries up as it rises
        self.costs = []      # (seconds, name) each module took, last frame

    def add_module(self, module, position=None, pull=False):
        if not hasattr(module, 'height') or module.height is None:
            raise ValueError(f"Module {module.__class__.__name__} must have a valid height attribute")
        self.modules.append((module, position))
        if pull:
            self.pulled.add(module)

    def set_scroll_speed(self, px_per_second):
        """Push a runtime scroll speed to every module that accepts one."""
        for module, _ in self.modules:
            setter = getattr(module, 'set_scroll_speed', None)
            if callable(setter):
                setter(px_per_second)

    def calculate_layout(self):
        positions = {}
        unspecified_modules = []
        for module, position in self.modules:
            if position is not None:
                positions[position] = module
            else:
                unspecified_modules.append(module)

        def try_place_modules(unspecified, gap_size):
            y_offset = 0
            placed_modules = []
            for position, module in sorted(positions.items()):
                print(f"Trying to place positioned module at {position} with height {module.height}")
                while unspecified and y_offset + unspecified[0].height + gap_size <= position:
                    module_to_place = unspecified.pop(0)
                    print(f"Placing unspecified module at {y_offset} with height {module_to_place.height}")
                    placed_modules.append((module_to_place, y_offset))
                    y_offset += module_to_place.height + gap_size
                if position + module.height > self.height:
                    print(f"Error: Positioned module at {position} with height {module.height} exceeds height {self.height}")
                    return None, unspecified
                placed_modules.append((module, position))
                y_offset = position + module.height
                if y_offset < self.height:
                    y_offset += gap_size

            # Place remaining unspecified modules
            while unspecified:
                module_to_place = unspecified.pop(0)
                if y_offset + module_to_place.height > self.height:
                    print(f"Error: Unspecified module at {y_offset} with height {module_to_place.height} exceeds height {self.height}")
                    return None, unspecified
                print(f"Placing remaining unspecified module at {y_offset} with height {module_to_place.height}")
                placed_modules.append((module_to_place, y_offset))
                y_offset += module_to_place.height
                if y_offset < self.height:
                    y_offset += gap_size

            # Check if final module placement uses exactly all available space
            if y_offset == self.height:
                print("Exact fit for all modules.")
                return placed_modules, []
            # Ensure it does not exceed the available space
            elif y_offset < self.height:
                print(f"Remaining space after placement: {self.height - y_offset} pixels")
                return placed_modules, []
            else:
                print(f"Error: Total height {y_offset} exceeds matrix height {self.height}")
                return None, unspecified

        def fill_gaps(gap_size):
            remaining_modules = unspecified_modules[:]
            while True:
                result = try_place_modules(remaining_modules, gap_size)
                if result is not None:
                    placed_modules, remaining_modules = result
                    if not remaining_modules:
                        break
                gap_size -= 1
                if gap_size < 0:
                    print(f"Error: Need an extra {remaining_modules[0].height} pixels.")
                    return None
            return placed_modules

        # Determine initial layout with 1-pixel gaps
        placed_modules = fill_gaps(1)

        if placed_modules:
            total_height = sum(module.height for module, _ in placed_modules)
            remaining_space = self.height - total_height
            print(f"Final total height: {total_height}, Remaining space: {remaining_space}")
            if total_height > self.height:
                return None

        return placed_modules

    def initialize_layout(self):
        if self.layout is None:
            self.layout = self.calculate_layout()
            if self.layout is None:
                raise ValueError("Insufficient space to place all modules")

    def pulled_rows(self):
        """(top, height) of each laid-out module the overview pulls with it."""
        self.initialize_layout()
        return [(y_offset, module.height) for module, y_offset in self.layout
                if module in self.pulled]

    def ticker(self):
        """The module scrolling text along the top row, if it can share it.

        The overview's title goes in the same place, and takes the line over
        from this module rather than cutting across it.
        """
        self.initialize_layout()
        for module, y_offset in self.layout:
            if (y_offset == 0 and callable(getattr(module, 'hand_over', None))
                    and callable(getattr(module, 'take_back', None))):
                return module
        return None

    def render(self):
        self.initialize_layout()

        final_image = Image.new('L', (self.width, self.height), 0)
        # What each module cost, for main.py to name the slow one when a frame
        # overruns. Two clock reads a module a frame is nothing beside drawing
        # one, and a gauge that waits on something is otherwise very hard to
        # tell apart from the panel being slow.
        self.costs = []
        for module, y_offset in self.layout:
            started = time.monotonic()
            module_image = module.render(self.width)
            self.costs.append((time.monotonic() - started,
                               module.__class__.__name__))
            if not isinstance(module_image, Image.Image):
                raise ValueError(f"Module {module.__class__.__name__} did not return a PIL Image")
            if module_image.size != (self.width, module.height):
                raise ValueError(f"Module {module.__class__.__name__} returned an image of size {module_image.size}, expected {(self.width, module.height)}")
            final_image.paste(module_image, (0, y_offset))

        return final_image

    def slowest(self):
        """The module that took longest last frame, as "Name 12ms"."""
        if not self.costs:
            return 'nothing'
        seconds, name = max(self.costs)
        return f'{name} {1000 * seconds:.0f}ms'
