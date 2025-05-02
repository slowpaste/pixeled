class MeteringUtility:
    def __init__(self, min_value, max_value, num_pixels, height=1):
        self.min_value = min_value
        self.max_value = max_value
        self.num_pixels = num_pixels
        self.height = height

    def render(self, width, values):
        # Check if values is a single number or a list
        if not isinstance(values, list):
            values = [values] * self.height  # Repeat the single value for each row
        
        # Create a 2D list representing the pixel matrix
        image = [[0 for _ in range(width)] for _ in range(self.height)]
        
        for row in range(self.height):
            value = values[row]
            scaled_value = (value - self.min_value) / (self.max_value - self.min_value) * self.num_pixels
            full_pixels = int(scaled_value)
            fractional_pixel = scaled_value - full_pixels
            
            for i in range(full_pixels):
                if i < width:  # Ensure we do not exceed the width
                    image[row][i] = 255  # Use 255 for a fully lit pixel (white in grayscale)
            
            if full_pixels < width:
                brightness = int(fractional_pixel * 255)
                image[row][full_pixels] = brightness
        
        return image
