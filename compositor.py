from PIL import Image

class Compositor:
    def __init__(self, width, height, config, mode=None):
        self.width = width
        self.height = height
        self.modules = []
        self.config = config
        self.mode = mode
        self.layout = None  # Store the layout after the initial placement

    def add_module(self, module, position=None, modes=None):
        if not hasattr(module, 'height') or module.height is None:
            raise ValueError(f"Module {module.__class__.__name__} must have a valid height attribute")
        self.modules.append((module, position, modes))
        self._tell_mode(module)

    def _tell_mode(self, module):
        """Let a module adapt to the display mode, if it cares.

        Frame rate differs by roughly 8x between modes, so a module animating
        per refresh rather than per second needs to know which one is running.
        """
        setter = getattr(module, 'set_mode', None)
        if callable(setter):
            setter(self.mode)

    def set_mode(self, mode):
        """Choose which display mode's layout to draw.

        Modules are built once and kept, so switching modes only changes which
        of them get laid out - rebuilding instead would restart the transit
        module's fetch threads on every toggle.
        """
        if mode != self.mode:
            self.mode = mode
            self.layout = None  # force a relayout on the next render
            for module, _, _ in self.modules:
                self._tell_mode(module)

    def set_scroll_speed(self, px_per_second):
        """Push a runtime scroll speed to every module that accepts one."""
        for module, _, _ in self.modules:
            setter = getattr(module, 'set_scroll_speed', None)
            if callable(setter):
                setter(px_per_second)

    def active_modules(self):
        """Modules that apply to the current mode.

        A module with no 'modes' list is drawn in every mode; otherwise it is
        drawn only in the modes it names.
        """
        return [(module, position) for module, position, modes in self.modules
                if not modes or self.mode is None or self.mode in modes]

    def calculate_layout(self):
        positions = {}
        unspecified_modules = []
        for module, position in self.active_modules():
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

    def render(self):
        self.initialize_layout()

        final_image = Image.new('L', (self.width, self.height), 0)
        for module, y_offset in self.layout:
            module_image = module.render(self.width)
            if not isinstance(module_image, Image.Image):
                raise ValueError(f"Module {module.__class__.__name__} did not return a PIL Image")
            if module_image.size != (self.width, module.height):
                raise ValueError(f"Module {module.__class__.__name__} returned an image of size {module_image.size}, expected {(self.width, module.height)}")
            final_image.paste(module_image, (0, y_offset))

        return final_image
