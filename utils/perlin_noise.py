import numpy as np

def perlin_noise(width, height, scale=10, seed=0, contrast=1.5):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    np.random.seed(seed)
    noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = j / scale
            y = i / scale
            x0, x1 = int(x), int(x) + 1
            y0, y1 = int(y), int(y) + 1
            sx, sy = f(x - x0), f(y - y0)
            n0 = np.dot((x - x0, y - y0), np.random.rand(2))
            n1 = np.dot((x - x1, y - y0), np.random.rand(2))
            ix0 = n0 + sx * (n1 - n0)
            n0 = np.dot((x - x0, y - y1), np.random.rand(2))
            n1 = np.dot((x - x1, y - y1), np.random.rand(2))
            ix1 = n0 + sx * (n1 - n0)
            noise[i][j] = ix0 + sy * (ix1 - ix0)

    # Normalize and adjust contrast
    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
    noise = ((noise - 0.5) * contrast + 0.2).clip(0, 1)

    return noise
