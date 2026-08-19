"""
Color analysis service for clustering and grouping similar colors.
Uses CIELAB color space for perceptually uniform clustering.
Uses MiniBatchKMeans for performance on large images.
"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

MAX_CLUSTER_PIXELS = 200_000


@dataclass
class ColorCluster:
    label: str
    representative_color_bgr: Tuple[int, int, int]
    representative_color_rgb: Tuple[int, int, int]
    pixel_count: int
    coverage_percent: float
    area_cm2: float
    mask: Optional[np.ndarray] = None
    visible: bool = True


def classify_color_name(bgr: Tuple[int, int, int]) -> str:
    b, g, r = bgr
    pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 30:
        return "BLACK"
    if s < 30 and v > 220:
        return "WHITE"
    if s < 30:
        if v < 100:
            return "DARK_GRAY"
        elif v < 180:
            return "GRAY"
        else:
            return "LIGHT_GRAY"

    if h < 10 or h >= 170:
        return "RED"
    elif h < 22:
        return "ORANGE"
    elif h < 33:
        return "YELLOW"
    elif h < 45:
        return "YELLOW_GREEN"
    elif h < 78:
        return "GREEN"
    elif h < 100:
        return "CYAN"
    elif h < 130:
        return "BLUE"
    elif h < 155:
        return "MAGENTA"
    else:
        return "RED"


def _bgr_to_lab(color_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert a single BGR color to 8-bit CIELAB as used by OpenCV."""
    pixel = np.uint8([[[color_bgr[0], color_bgr[1], color_bgr[2]]]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0][0]
    return (int(lab[0]), int(lab[1]), int(lab[2]))


def _color_lab_distance(
    color_bgr: Tuple[int, int, int],
    lab_b: Tuple[int, int, int],
) -> float:
    """CIE76 (Euclidean) distance between a BGR color and a LAB color.

    Distances are computed in 8-bit CIELAB (L, a, b each 0-255), which is
    perceptually far more meaningful than raw BGR channel differences: two
    visually similar colors get a small distance and two visually different
    colors get a large one. This is used only for background detection; the
    color clustering itself is untouched.
    """
    lab_a = np.array(_bgr_to_lab(color_bgr), dtype=np.float32)
    lab_b = np.array(lab_b, dtype=np.float32)
    return float(np.sqrt(np.sum((lab_a - lab_b) ** 2)))


class ColorAnalyzer:

    def __init__(self):
        self.color_space = "lab"

    def prepare_features(
        self, roi_bgr: np.ndarray, color_space: str = "lab"
    ) -> np.ndarray:
        if color_space == "lab":
            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2Lab)
            return lab.reshape(-1, 3).astype(np.float32)
        elif color_space == "hsv":
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            features = hsv.reshape(-1, 3).astype(np.float32)
            # OpenCV stores Hue in 0-179 while Saturation and Value use
            # 0-255.  To prevent KMeans from over-weighting the Hue channel
            # (whose range is ~70% of the other two), we scale Hue to
            # 0-89.5 by dividing by 2 so all three channels occupy a
            # similar numeric range during clustering.
            #
            # IMPORTANT: This transformation MUST be reversed in
            # get_representative_colors() (multiply by 2) before the
            # cluster centers are converted back to BGR via OpenCV.
            # Without the reversal a blue pixel (H≈120) would be
            # clustered as H≈60 and decoded as green.
            features[:, 0] *= 0.5
            return features
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

    def _downsample_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Downsample large images so KMeans runs on at most MAX_CLUSTER_PIXELS."""
        h, w = image.shape[:2]
        total = h * w
        if total <= MAX_CLUSTER_PIXELS:
            return image, 1.0
        scale = np.sqrt(MAX_CLUSTER_PIXELS / total)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def cluster_colors(
        self,
        roi_bgr: np.ndarray,
        n_colors: int = 5,
        color_space: str = "lab",
        max_iter: int = 100,
        n_init: int = 3,
    ) -> Tuple[np.ndarray, MiniBatchKMeans]:
        features = self.prepare_features(roi_bgr, color_space)

        kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=42,
            max_iter=max_iter,
            n_init=n_init,
            batch_size=min(1024, len(features)),
        )
        labels = kmeans.fit_predict(features)

        return labels.reshape(roi_bgr.shape[:2]), kmeans

    def get_representative_colors(
        self, kmeans, color_space: str = "lab"
    ) -> List[Tuple[int, int, int]]:
        # Work on a copy so the KMeans model's centers are not mutated.
        centers = kmeans.cluster_centers_.copy()

        if color_space == "lab":
            # CIELAB centers are already in 8-bit OpenCV range (L, a, b all
            # 0-255), so they can be cast straight to uint8 and converted.
            centers_uint8 = centers.astype(np.uint8).reshape(-1, 1, 3)
            bgr_centers = cv2.cvtColor(centers_uint8, cv2.COLOR_Lab2BGR)
            return [tuple(int(v) for v in c[0]) for c in bgr_centers]
        elif color_space == "hsv":
            # Reverse the Hue halving applied in prepare_features().
            # The cluster centers are in the scaled space (H/2, S, V)
            # so we multiply Hue by 2 to restore real OpenCV values.
            # Without this, e.g. a blue with H≈120 would be clustered
            # as H≈60 and decoded back as green.
            centers[:, 0] *= 2.0
            # Clip each channel to its valid OpenCV HSV range:
            #   H: 0–179,  S: 0–255,  V: 0–255
            # A uniform clip to 255 is NOT correct because Hue values
            # above 179 would be interpreted as different hues by
            # cv2.COLOR_HSV2BGR (wrapping or producing wrong colors).
            centers = np.clip(centers, [0, 0, 0], [179, 255, 255])
            centers_uint8 = centers.astype(np.uint8).reshape(-1, 1, 3)
            bgr_centers = cv2.cvtColor(centers_uint8, cv2.COLOR_HSV2BGR)
            return [tuple(int(v) for v in c[0]) for c in bgr_centers]

    def create_segmentation_preview(
        self,
        roi_bgr: np.ndarray,
        labels: np.ndarray,
        representative_colors: List[Tuple[int, int, int]],
    ) -> np.ndarray:
        h, w = labels.shape
        seg_map = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(representative_colors):
            mask = labels == i
            seg_map[mask] = color
        return seg_map

    def analyze_colors(
        self,
        roi_bgr: np.ndarray,
        n_colors: int = 5,
        color_space: str = "lab",
        background_color: Optional[Tuple[int, int, int]] = None,
        background_tolerance: int = 30,
    ) -> Dict:
        h, w = roi_bgr.shape[:2]
        total_pixels = h * w

        small_img, scale = self._downsample_image(roi_bgr)
        labels_small, kmeans = self.cluster_colors(
            small_img, n_colors, color_space
        )

        if scale < 1.0:
            labels = cv2.resize(
                labels_small, (w, h), interpolation=cv2.INTER_NEAREST
            )
        else:
            labels = labels_small

        rep_colors = self.get_representative_colors(kmeans, color_space)
        seg_preview = self.create_segmentation_preview(roi_bgr, labels, rep_colors)

        # Background is supplied (or auto-detected) in BGR. Convert it to
        # CIELAB once so every cluster comparison below is perceptual.
        bg_lab = None
        if background_color is not None:
            bg_lab = _bgr_to_lab(background_color)

        clusters = []
        unique_labels, counts = np.unique(labels, return_counts=True)

        sort_idx = np.argsort(-counts)
        unique_labels = unique_labels[sort_idx]
        counts = counts[sort_idx]

        for label_id, count in zip(unique_labels, counts):
            label_id = int(label_id)
            count = int(count)
            bgr = rep_colors[label_id]
            name = classify_color_name(bgr)

            is_background = False
            if bg_lab is not None:
                # Compare in CIELAB instead of BGR: the LAB distance matches
                # how humans perceive color differences, so the same tolerance
                # value now cleanly separates a background from printed colors
                # that merely *look* different from it. The tolerance is in
                # 8-bit LAB units (each channel 0-255); the default 30 is a
                # moderate perceptual threshold.
                diff = _color_lab_distance(bgr, bg_lab)
                is_background = diff < background_tolerance

            cluster = ColorCluster(
                label=f"{name}_{label_id}" if not is_background else "BACKGROUND",
                representative_color_bgr=tuple(int(v) for v in bgr),
                representative_color_rgb=tuple(int(v) for v in (bgr[2], bgr[1], bgr[0])),
                pixel_count=count,
                coverage_percent=float((count / total_pixels) * 100),
                area_cm2=0.0,
                mask=self._create_mask(labels, label_id, (h, w)),
                visible=True,
            )
            clusters.append(cluster)

        bg_clusters = [c for c in clusters if c.label == "BACKGROUND"]
        non_bg_clusters = [c for c in clusters if c.label != "BACKGROUND"]
        clusters = non_bg_clusters + bg_clusters

        clusters = self._merge_duplicate_names(clusters, total_pixels)

        return {
            "clusters": clusters,
            "labels": labels,
            "segmentation_preview": seg_preview,
            "representative_colors": rep_colors,
            "total_pixels": total_pixels,
        }

    def _create_mask(self, labels: np.ndarray, cluster_id: int, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[labels == cluster_id] = 255
        return mask

    def _merge_duplicate_names(
        self, clusters: List[ColorCluster], total_pixels: int
    ) -> List[ColorCluster]:
        name_map = {}
        merged = []

        for c in clusters:
            base_name = c.label.split("_")[0] if "_" in c.label else c.label

            if c.label == "BACKGROUND":
                merged.append(c)
                continue

            if base_name not in name_map:
                name_map[base_name] = c
            else:
                existing = name_map[base_name]
                existing.pixel_count += c.pixel_count
                existing.coverage_percent = (existing.pixel_count / total_pixels) * 100
                if existing.mask is not None and c.mask is not None:
                    existing.mask = cv2.bitwise_or(existing.mask, c.mask)

        for name, cluster in name_map.items():
            merged.append(cluster)

        return merged
