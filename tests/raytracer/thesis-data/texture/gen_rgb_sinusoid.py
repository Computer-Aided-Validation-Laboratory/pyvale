import numpy as np
from PIL import Image
import argparse
import os

def generate_sinusoidal_rgb(input_path, output_path, period_pct):
    # Load original speckle texture
    try:
        img = Image.open(input_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    data = np.array(img, dtype=np.float32)
    h, w, c = data.shape
    
    # Calculate period in pixels
    period_px = (period_pct / 100.0) * w
    freq = 2 * np.pi / period_px

    # Create coordinate grids
    y, x = np.ogrid[0:h, 0:w]

    # 1. Blue sinusoid: Left to right (Channel 2)
    # sin range [-1, 1] -> map to some intensity added to current
    # We'll use a multiplicative or additive approach? 
    # Let's go with adding a pattern and clipping.
    blue_sin = (np.sin(freq * x) + 1.0) / 2.0 * 127.0
    data[:, :, 2] += blue_sin

    # 2. Red sinusoid: Top to bottom (Channel 0)
    red_sin = (np.sin(freq * y) + 1.0) / 2.0 * 127.0
    data[:, :, 0] += red_sin

    # 3. Green sinusoid: Bottom-left to top-right (Channel 1)
    # Vector from (h, 0) to (0, w)
    # Distance along this vector:
    # Projection of (x, y) onto the direction (w, -h)
    # But let's just use a simple diagonal coordinate: x + (h - y)
    diag_coord = x + (h - 1 - y)
    green_sin = (np.sin(freq * diag_coord) + 1.0) / 2.0 * 127.0
    data[:, :, 1] += green_sin

    # Clip and convert back to uint8
    data = np.clip(data, 0, 255).astype(np.uint8)
    
    result_img = Image.fromarray(data)
    result_img.save(output_path)
    print(f"Saved RGB texture to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add sinusoidal RGB patterns to a texture.")
    parser.add_argument("--input", default="texture/speckle.tiff", help="Input image path")
    parser.add_argument("--output", default="texture/speckle_rgb.bmp", help="Output image path")
    parser.add_argument("--period", type=float, default=20.0, help="Period as percentage of width")
    
    args = parser.parse_args()
    
    generate_sinusoidal_rgb(args.input, args.output, args.period)
