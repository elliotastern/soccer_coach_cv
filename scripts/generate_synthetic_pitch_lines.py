#!/usr/bin/env python3
"""
Lightweight two-pass synthetic data for pitch line segmentation (hallucination training).
Pass 1: Draw pitch lines on green canvas, overlay occluders -> RGB image (input X).
Pass 2: Same canvas without occluders -> binary line mask (target Y).
Output: image_dir + mask_dir compatible with train_pitch_line_segmentation.py.
"""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np


# FIFA pitch in "canvas" units (e.g. 1050 x 680); origin top-left
PITCH_LENGTH = 1050
PITCH_WIDTH = 680
LINE_WIDTH = 8
CENTER_CIRCLE_R = 91
PENALTY_BOX_DEPTH = 165
PENALTY_BOX_WIDTH = 403
GOAL_AREA_DEPTH = 55
GOAL_AREA_WIDTH = 183
PENALTY_SPOT_DIST = 110


def draw_pitch_lines(mask: np.ndarray, ox: int, oy: int, scale: float, line_width: int, value: int = 255):
    """Draw FIFA pitch lines on mask. ox,oy = origin; scale = pixels per unit."""
    w = int(PITCH_LENGTH * scale)
    h = int(PITCH_WIDTH * scale)
    # Outline
    cv2.rectangle(mask, (ox, oy), (ox + w, oy + h), value, line_width)
    # Halfway line
    cv2.line(mask, (ox + w // 2, oy), (ox + w // 2, oy + h), value, line_width)
    # Center circle
    cx, cy = ox + w // 2, oy + h // 2
    cv2.circle(mask, (cx, cy), int(CENTER_CIRCLE_R * scale), value, line_width)
    # Left penalty box
    cv2.rectangle(mask, (ox, oy + (h - int(PENALTY_BOX_WIDTH * scale)) // 2),
                  (ox + int(PENALTY_BOX_DEPTH * scale), oy + (h + int(PENALTY_BOX_WIDTH * scale)) // 2),
                  value, line_width)
    # Left goal area
    cv2.rectangle(mask, (ox, oy + (h - int(GOAL_AREA_WIDTH * scale)) // 2),
                  (ox + int(GOAL_AREA_DEPTH * scale), oy + (h + int(GOAL_AREA_WIDTH * scale)) // 2),
                  value, line_width)
    # Left penalty spot
    cv2.circle(mask, (ox + int(PENALTY_SPOT_DIST * scale), oy + h // 2), line_width * 2, value, -1)
    # Right penalty box
    cv2.rectangle(mask, (ox + w - int(PENALTY_BOX_DEPTH * scale), oy + (h - int(PENALTY_BOX_WIDTH * scale)) // 2),
                  (ox + w, oy + (h + int(PENALTY_BOX_WIDTH * scale)) // 2), value, line_width)
    # Right goal area
    cv2.rectangle(mask, (ox + w - int(GOAL_AREA_DEPTH * scale), oy + (h - int(GOAL_AREA_WIDTH * scale)) // 2),
                  (ox + w, oy + (h + int(GOAL_AREA_WIDTH * scale)) // 2), value, line_width)
    # Right penalty spot
    cv2.circle(mask, (ox + w - int(PENALTY_SPOT_DIST * scale), oy + h // 2), line_width * 2, value, -1)


def generate_one_pair(
    out_width: int,
    out_height: int,
    num_occluders: int,
    occluder_max_scale: float,
    line_width_range: tuple,
    green_low: tuple,
    green_high: tuple,
    rng: random.Random,
) -> tuple:
    """
    Generate one (image_with_occluders, mask_lines_only).
    Image has pitch + lines + occluders; mask has lines only (no occluders).
    """
    # Green background (BGR) with slight variation
    b = rng.randint(green_low[0], green_high[0])
    g = rng.randint(green_low[1], green_high[1])
    r = rng.randint(green_low[2], green_high[2])
    image = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    image[:] = (b, g, r)
    mask = np.zeros((out_height, out_width), dtype=np.uint8)

    scale = min((out_width - 40) / PITCH_LENGTH, (out_height - 40) / PITCH_WIDTH)
    ox = (out_width - int(PITCH_LENGTH * scale)) // 2
    oy = (out_height - int(PITCH_WIDTH * scale)) // 2
    lw = rng.randint(line_width_range[0], line_width_range[1])

    # Draw lines on both image and mask (white lines on green)
    draw_pitch_lines(mask, ox, oy, scale, lw, 255)
    draw_pitch_lines(image, ox, oy, scale, lw, (255, 255, 255))

    # Occluders: rectangles that cover parts of lines (simulate players)
    for _ in range(num_occluders):
        bw = int(out_width * occluder_max_scale * (0.03 + rng.random() * 0.08))
        bh = int(out_height * occluder_max_scale * (0.08 + rng.random() * 0.15))
        x = rng.randint(0, max(0, out_width - bw))
        y = rng.randint(0, max(0, out_height - bh))
        color = (b, g, r)  # Same as background so they "occlude" lines
        cv2.rectangle(image, (x, y), (x + bw, y + bh), color, -1)
        # Do NOT draw on mask -> target mask has lines only (hallucination target)

    return image, mask


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pitch line images for hallucination training")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_pitch_lines",
                        help="Output directory (will create images/ and masks/ inside)")
    parser.add_argument("--num_images", type=int, default=500, help="Number of image/mask pairs")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--num_occluders", type=int, default=8, help="Occluders per image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    green_low = (80, 120, 80)
    green_high = (120, 200, 120)
    line_width_range = (3, 10)

    for i in range(args.num_images):
        image, mask = generate_one_pair(
            args.width, args.height,
            num_occluders=args.num_occluders,
            occluder_max_scale=1.0,
            line_width_range=line_width_range,
            green_low=green_low,
            green_high=green_high,
            rng=rng,
        )
        name = f"syn_{i:05d}.png"
        cv2.imwrite(str(img_dir / name), image)
        cv2.imwrite(str(mask_dir / name), mask)

    print(f"Generated {args.num_images} pairs in {out_dir}")
    print(f"  images: {img_dir}")
    print(f"  masks:  {mask_dir}")
    print("Train with: python scripts/train_pitch_line_segmentation.py --train_images ... --train_masks ...")


if __name__ == "__main__":
    main()
