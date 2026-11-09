"""Data augmentation for Arabic OCR.

Smoke-test mode: every operation is a no-op that returns the input image
unchanged. Each method is a drop-in stub with a TODO for the real
implementation so the pipeline can be exercised end-to-end first.
"""

import random

from PIL import Image, ImageEnhance, ImageFilter


class RandomAugment:
    """Random augmentation pipeline for OCR images.

    Args:
        p: probability of applying each augmentation (unused in smoke test).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        # Smoke test: no-op, return input unchanged
        return img

    def random_crop(self, img):
        # TODO: implement random crop (e.g. 90-100% of width/height)
        return img

    def random_rotation(self, img):
        # TODO: rotate by a random angle in [-5, 5] degrees
        return img

    def random_scale(self, img):
        # TODO: scale by a random factor in [0.9, 1.1]
        return img

    def gaussian_noise(self, img):
        # TODO: inject zero-mean Gaussian noise
        return img

    def gaussian_blur(self, img):
        # TODO: apply Gaussian blur with a small random radius
        return img

    def adjust_contrast(self, img):
        # TODO: adjust contrast by a random factor
        return img
