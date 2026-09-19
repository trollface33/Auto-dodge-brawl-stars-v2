"""Utility to render a debug overlay for the V2 detector."""

import os
from datetime import datetime

from PIL import Image, ImageDraw


def save_debug_frame(frame, player_position, candidates, threat, dodge_target=None, output_dir="debug"):
    """Save an annotated screenshot showing the threat and dodge target."""
    os.makedirs(output_dir, exist_ok=True)
    image = frame.copy()
    draw = ImageDraw.Draw(image)

    width, height = image.size
    roi_left = int(width * 0.08)
    roi_top = int(height * 0.10)
    roi_right = int(width * 0.92)
    roi_bottom = int(height * 0.92)
    draw.rectangle([roi_left, roi_top, roi_right, roi_bottom], outline=(0, 255, 255), width=2)

    for candidate in candidates or []:
        x, y = candidate["position"]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 0, 0), width=2)

    if player_position:
        px, py = player_position
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), outline=(0, 0, 255), width=3)
        draw.text((px + 14, py - 18), "PLAYER", fill=(0, 0, 255))

    if threat:
        tx, ty = threat
        draw.ellipse((tx - 10, ty - 10, tx + 10, ty + 10), outline=(0, 255, 0), width=3)
        draw.text((tx + 14, ty + 14), "THREAT", fill=(0, 255, 0))

    if dodge_target:
        dx, dy = dodge_target
        draw.line([player_position[0], player_position[1], dx, dy], fill=(255, 255, 0), width=3)
        draw.ellipse((dx - 12, dy - 12, dx + 12, dy + 12), outline=(255, 255, 0), width=3)
        draw.text((dx + 14, dy + 14), "DODGE", fill=(255, 255, 0))

    filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path
