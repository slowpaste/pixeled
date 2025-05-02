import numpy as np
from PIL import Image, ImageEnhance
from modules.module_base import ModuleBase
from utils.perlin_noise import perlin_noise

class NoiseModule(ModuleBase):
    def __init__(self, height):
        super().__init__(height)
        self.width = 9
        self.high_res_width = self.width * 2
        self.high_res_height = self.height * 2
        self.noise1 = perlin_noise(self.high_res_width, self.high_res_height, scale=10, seed=42, contrast=1.5)
        self.noise2 = perlin_noise(self.high_res_width, self.high_res_height, scale=10, seed=43, contrast=1.5)
        self.offset = 0
        self.velocity = 0.0
        self.target_velocity = 0.0
        self.brightness_factor = 1.0
        self.target_brightness_factor = 1.0
        self.acceleration = 0.2
        self.brightness_change_rate = 0.1
        self.status = "Not charging"
        self.seed_counter = 44  # Counter to generate new seeds

        # Adjust the brightness of the first and last rows
        self.noise1[0] *= 0.5
        self.noise1[-1] *= 0.5
        self.noise2[0] *= 0.5
        self.noise2[-1] *= 0.5

    def get_battery_status(self):
        with open('/sys/class/power_supply/BAT1/status', 'r') as f:
            return f.read().strip()

    def generate_noise_block(self, seed):
        noise_block = perlin_noise(self.high_res_width, self.high_res_height, scale=10, seed=seed, contrast=1.5)
        noise_block[0] *= 0.5
        noise_block[-1] *= 0.5
        return noise_block

    def adjust_brightness(self, image, factor):
        return ImageEnhance.Brightness(image).enhance(factor)

    def adjust_contrast(self, image, factor):
        return ImageEnhance.Contrast(image).enhance(factor)

    def update_velocity_and_brightness(self):
        if self.velocity < self.target_velocity:
            self.velocity = min(self.velocity + self.acceleration, self.target_velocity)
        elif self.velocity > self.target_velocity:
            self.velocity = max(self.velocity - self.acceleration, self.target_velocity)

        if self.brightness_factor < self.target_brightness_factor:
            self.brightness_factor = min(self.brightness_factor + self.brightness_change_rate, self.target_brightness_factor)
        elif self.brightness_factor > self.target_brightness_factor:
            self.brightness_factor = max(self.brightness_factor - self.brightness_change_rate, self.target_brightness_factor)

    def render(self, width):
        self.status = self.get_battery_status()

        # Update target velocity and brightness based on status
        if self.status == "Charging":
            self.target_velocity = 2  # Scroll left
            self.target_brightness_factor = 1.0  # Full brightness
        elif self.status == "Discharging":
            self.target_velocity = -2  # Scroll right
            self.target_brightness_factor = 0.5  # Half brightness
        else:  # Not charging
            self.target_velocity = 0.0  # Stop scrolling
            self.target_brightness_factor = 0.5  # Half brightness

        # Update current velocity and brightness factor
        self.update_velocity_and_brightness()

        # Calculate new offset
        self.offset += self.velocity
        if self.offset >= self.high_res_width:
            self.offset -= self.high_res_width
            self.noise1 = self.noise2
            self.noise2 = self.generate_noise_block(self.seed_counter)
            self.seed_counter += 1
        elif self.offset < 0:
            self.offset += self.high_res_width
            self.noise1 = self.noise2
            self.noise2 = self.generate_noise_block(self.seed_counter)
            self.seed_counter += 1

        # Create combined noise image with higher resolution
        combined_noise = np.hstack((self.noise1, self.noise2))
        int_offset = int(self.offset)
        noise_image = Image.fromarray(np.uint8(combined_noise[:, int_offset:int_offset + self.high_res_width] * 255))
        noise_image = self.adjust_brightness(noise_image, self.brightness_factor)

        # Downsample to display resolution
        noise_image = noise_image.resize((self.width, self.height), Image.BILINEAR)

        # Adjust contrast after downsampling
        noise_image = self.adjust_contrast(noise_image, 1.5)

        image = super().render(width)
        image.paste(noise_image, (0, 0))
        return image
