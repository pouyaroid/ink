"""
Image processing service for ROI extraction and preprocessing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class ImageProcessor:
    """Handles image loading, ROI extraction, and preprocessing."""

    def __init__(self):
        self.gaussian_kernel = (5, 5)
        self.median_kernel = 5

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image from disk."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        return image

    def extract_roi(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract region of interest from the image."""
        h_img, w_img = image.shape[:2]

        x = max(0, min(x, w_img))
        y = max(0, min(y, h_img))
        x2 = max(0, min(x + w, w_img))
        y2 = max(0, min(y + h, h_img))

        if x2 <= x or y2 <= y:
            raise ValueError(f"Invalid ROI: ({x},{y}) to ({x2},{y2})")

        return image[y:y2, x:x2].copy()

    def preprocess(
        self,
        roi: np.ndarray,
        apply_gaussian: bool = True,
        apply_median: bool = False,
        gaussian_ksize: int = 5,
        median_ksize: int = 5,
    ) -> np.ndarray:
        """Apply noise reduction preprocessing to ROI."""
        processed = roi.copy()

        if apply_gaussian and gaussian_ksize > 1:
            k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
            processed = cv2.GaussianBlur(processed, (k, k), 0)

        if apply_median and median_ksize > 1:
            k = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
            processed = cv2.medianBlur(processed, k)

        return processed

    def convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to CIELAB color space."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    def convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to HSV color space."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_image_dimensions(self, image: np.ndarray) -> dict:
        """Return image height and width."""
        h, w = image.shape[:2]
        return {"width": w, "height": h}

    def create_mask_from_cluster(
        self, labels: np.ndarray, cluster_id: int, shape: tuple
    ) -> np.ndarray:
        """Create a binary mask for a specific cluster."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        mask[labels == cluster_id] = 255
        return mask

    def apply_morphological_cleanup(
        self, mask: np.ndarray, kernel_size: int = 3
    ) -> np.ndarray:
        """Apply morphological operations to clean up masks."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cleaned
