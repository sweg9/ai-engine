"""Plain NumPy helpers on images as OpenCV gives them: uint8, height x width x 3, BGR."""

import numpy as np


def check_image(img, name="image"):
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must be a uint8 array of height x width x 3 (BGR)")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"{name} is empty")
    return img


def check_box(box, name="box"):
    if len(box) != 4 or not all(0.0 <= float(v) <= 1.0 for v in box):
        raise ValueError(f"{name} must be [x1, y1, x2, y2] with ratios from 0 to 1, got {box}")
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} corners are out of order: {box}")
    return x1, y1, x2, y2


def gray(img):
    """Luminance from 0 to 1 (BGR weights of ITU-R BT.601)."""
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32) / 255


def crop(img, box, margin=0.0):
    """The part of `img` inside `box`, grown by `margin` times its size on each side.
    Never empty: at least one pixel."""
    x1, y1, x2, y2 = check_box(box)
    h, w = img.shape[:2]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    c0, r0 = int(max(0.0, x1 - mx) * w), int(max(0.0, y1 - my) * h)
    c1, r1 = int(np.ceil(min(1.0, x2 + mx) * w)), int(np.ceil(min(1.0, y2 + my) * h))
    c0, r0 = min(c0, w - 1), min(r0, h - 1)
    return img[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]


def resize(a, size):
    """Nearest-neighbour resize of a 2-D array to (height, width)."""
    h, w = a.shape[:2]
    rows = (np.arange(size[0]) * h // size[0]).clip(0, h - 1)
    cols = (np.arange(size[1]) * w // size[1]).clip(0, w - 1)
    return a[rows][:, cols]


def sharpness(img):
    """Variance of the Laplacian of the gray image, on a 0 to 255 scale."""
    g = gray(img) * 255
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())
