import time

import numpy as np
from PIL import Image, ImageEnhance
from modules.module_base import ModuleBase
from perlin_noise import PerlinNoise

class PrecomputedNoise(ModuleBase):
    def __init__(self, height, scroll_speed=2.0):
        super().__init__(height)
        self.width = 9
        self.high_res_width = self.width * 32  # Making the noise image 6 times wider than the display width
        self.high_res_height = self.height * 2
        self.noise = PerlinNoise(octaves=5, seed=42)
        self.noise_image = self.generate_cylindrical_noise_texture(self.high_res_width, self.high_res_height, scale=1.5)
        # Scroll rate in display pixels per SECOND, not per frame. The panel runs
        # at ~6fps on stock firmware but ~60fps on the patched build, so a
        # per-frame step would scroll 10x faster on one than the other. Sub-pixel offsets
        # blend between adjacent texture columns, so slow motion still reads as
        # continuous rather than stepping a whole pixel at a time.
        self.scroll_speed = scroll_speed
        self.velocity = 0.0
        self.target_velocity = 0.0
        self.brightness_factor = 1.0
        self.target_brightness_factor = 1.0
        self.acceleration = scroll_speed * 1.2   # ramp to full speed in ~0.85s
        self.brightness_change_rate = 0.6        # per second
        self._last_render = None
        self.status = "Not charging"
        self.offset = 0

        # Adjust the brightness of the first and last rows
        self.noise_image = self.noise_image.astype(np.float64)  # Convert to float64 for multiplication
        self.noise_image[0] *= 0.2
        self.noise_image[-1] *= 0.2
        self.noise_image = self.noise_image.astype(np.uint8)  # Convert back to uint8

        # Save the texture as an image (optional)
        # self.save_texture_image()

    def generate_cylindrical_noise_texture(self, width, height, scale):
        texture = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                u = j / float(width)
                v = i / float(height)
                theta = 2 * np.pi * u
                y = v * height
                x = np.cos(theta)
                z = np.sin(theta)
                noise_value = self.noise([x * scale, y * scale, z * scale])
                texture[i, j] = noise_value

        # Normalize the texture to [0, 1]
        texture = (texture - texture.min()) / (texture.max() - texture.min())

        # Apply gamma correction
        gamma = 0.4
        texture = np.power(texture, 1/gamma)

        # Scale to [0, 255]
        texture = (texture * 255).astype(np.uint8)
        return texture

    def save_texture_image(self):
        # Save the generated texture as an image for visualization or debugging
        texture_image = Image.fromarray(self.noise_image)  # The array is already in [0, 255] range
        texture_image.save("noise_texture.png")
        texture_image.show()

    def adjust_brightness(self, image, factor):
        return ImageEnhance.Brightness(image).enhance(factor)

    def adjust_contrast(self, image, factor):
        return ImageEnhance.Contrast(image).enhance(factor)

    def get_battery_status(self):
        with open('/sys/class/power_supply/BAT1/status', 'r') as f:
            return f.read().strip()

    def update_velocity_and_brightness(self, dt):
        accel = self.acceleration * dt
        if self.velocity < self.target_velocity:
            self.velocity = min(self.velocity + accel, self.target_velocity)
        elif self.velocity > self.target_velocity:
            self.velocity = max(self.velocity - accel, self.target_velocity)

        rate = self.brightness_change_rate * dt
        if self.brightness_factor < self.target_brightness_factor:
            self.brightness_factor = min(self.brightness_factor + rate, self.target_brightness_factor)
        elif self.brightness_factor > self.target_brightness_factor:
            self.brightness_factor = max(self.brightness_factor - rate, self.target_brightness_factor)

    def render(self, width):
        self.status = self.get_battery_status()

        # Update target velocity and brightness based on status
        if self.status == "Charging":
            self.target_velocity = self.scroll_speed  # Scroll left
            self.target_brightness_factor = 1.0  # Full brightness
        elif self.status == "Discharging":
            self.target_velocity = -self.scroll_speed  # Scroll right
            self.target_brightness_factor = 0.5  # Half brightness
        else:  # Not charging
            self.target_velocity = 0.0  # Stop scrolling
            self.target_brightness_factor = 0.5  # Half brightness

        # Elapsed wall time since the last frame. Clamped so a stall (startup,
        # config reload) can't jump the scroll a long way.
        now = time.monotonic()
        dt = 0.0 if self._last_render is None else min(now - self._last_render, 0.25)
        self._last_render = now

        # Update current velocity and brightness factor
        self.update_velocity_and_brightness(dt)

        # Calculate new offset
        self.offset += self.velocity * dt
        self.offset %= self.high_res_width  # Ensure the offset wraps around correctly

        # Sample the texture at the fractional offset, blending the two adjacent
        # integer slices. This is what makes sub-pixel steps visible: the grey
        # levels themselves carry the fraction of a pixel we've moved, so motion
        # reads as continuous rather than stepping a whole pixel at a time.
        # np.take(mode='wrap') also handles the cylindrical wrap for free.
        int_offset = int(self.offset)
        frac = self.offset - int_offset
        columns = np.arange(int_offset, int_offset + self.width)
        near = np.take(self.noise_image, columns, axis=1, mode='wrap').astype(np.float64)
        far = np.take(self.noise_image, columns + 1, axis=1, mode='wrap').astype(np.float64)
        blended = near * (1.0 - frac) + far * frac

        noise_image = Image.fromarray(blended.astype(np.uint8))
        noise_image = self.adjust_brightness(noise_image, self.brightness_factor)

        # Downsample to display resolution
        noise_image = noise_image.resize((self.width, self.height), Image.BILINEAR)

        # Adjust contrast after downsampling
        noise_image = self.adjust_contrast(noise_image, 1.5)

        image = super().render(width)
        image.paste(noise_image, (0, 0))
        return image

# Example of generating and displaying the noise texture
if __name__ == "__main__":
    width, height, scale = 18, 8, 10
    module = NoiseModule(height)
