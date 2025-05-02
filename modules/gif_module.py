import os
import numpy as np
from PIL import Image, ImageSequence
from modules.module_base import ModuleBase
import time

class GifModule(ModuleBase):
    def __init__(self, height, gif_path='assets/smalleye.gif'):
        super().__init__(height)
        self.gif_path = gif_path
        self.frames = self.load_gif_frames()
        self.current_frame = 0
        self.last_update_time = time.time()
        self.frame_duration = 0.3  # Adjust this value for desired frame rate

    def load_gif_frames(self):
        if not os.path.exists(self.gif_path):
            raise FileNotFoundError(f"GIF file not found: {self.gif_path}")

        gif = Image.open(self.gif_path)
        frames = []

        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert("L").resize((9, self.height), Image.BILINEAR)
            frames.append(np.array(frame))

        return frames

    def render(self, width):
        current_time = time.time()

        # Update the frame index if enough time has passed
        if current_time - self.last_update_time >= self.frame_duration:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.last_update_time = current_time

        image = super().render(width)

        # Get the current frame
        frame = self.frames[self.current_frame]

        # Convert the frame back to PIL Image for pasting
        frame_image = Image.fromarray(frame)

        # Paste the frame onto the image
        image.paste(frame_image, (0, 0))

        return image
