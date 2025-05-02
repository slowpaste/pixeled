class GaugesModule(ModuleBase):
    def __init__(self, height=13, width=9):
        super().__init__(height)
        self.width = width
        self.fuel = 1.0
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.speed = 0.0

    def update_values(self, fuel, throttle, brake, clutch, speed):
        self.fuel = fuel
        self.throttle = throttle
        self.brake = brake
        self.clutch = clutch
        self.speed = speed

    def _value_to_bar(self, ratio, height):
        ratio_clamped = max(0.0, min(1.0, ratio))
        filled_length = int(ratio_clamped * height)
        bar = [255 if y < filled_length else 0 for y in range(height)]
        return bar

    def render(self, width):
        image = Image.new('L', (width, self.height), 0)

        # Define gauge positions
        fuel_bar = self._value_to_bar(self.fuel, self.height)
        throttle_bar = self._value_to_bar(self.throttle, self.height)
        brake_bar = self._value_to_bar(self.brake, self.height)
        clutch_bar = self._value_to_bar(self.clutch, self.height)

        # Assign each gauge to a specific column
        gauges = [fuel_bar, throttle_bar, brake_bar, clutch_bar]

        # Define columns for each gauge
        # For example:
        # Fuel: column 1
        # Throttle: column 3
        # Brake: column 5
        # Clutch: column 7

        gauge_columns = [1, 3, 5, 7]
        for gauge, col in zip(gauges, gauge_columns):
            for y, val in enumerate(gauge):
                image.putpixel((col, self.height - y - 1), val)  # Bottom-up

        # Optional: Display speed as additional information or use remaining columns

        return image
