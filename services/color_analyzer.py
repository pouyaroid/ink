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


def _bgr_to_lab(color_bgr: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert a single BGR color to floating-point CIELAB.

    Floating point conversion yields true CIE L* (0-100) and a*/b*
    (-127..127). OpenCV's 8-bit Lab encoding instead scales L by 255/100
    while leaving a/b as shifted bytes (+128), so Euclidean distances in
    8-bit Lab amplify lightness differences by ~2.55x relative to chromatic
    ones. Using real CIELAB coordinates keeps distances perceptually valid.
    """
    pixel = np.array(
        [[[color_bgr[0], color_bgr[1], color_bgr[2]]]], dtype=np.float32
    ) / 255.0
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0][0]
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def _color_lab_distance(
    color_bgr: Tuple[int, int, int],
    lab_b: Tuple[float, float, float],
) -> float:
    """CIE76 (Euclidean) Delta-E between a BGR color and a LAB color.

    Distances are computed in floating-point CIELAB, i.e. this is a true
    CIE76 Delta-E where ~1.0 is one just-noticeable difference and 100 is
    roughly the distance between black and white. This is used only for
    background detection; the color clustering itself is untouched.
    """
    lab_a = np.array(_bgr_to_lab(color_bgr), dtype=np.float32)
    lab_b = np.array(lab_b, dtype=np.float32)
    return float(np.sqrt(np.sum((lab_a - lab_b) ** 2)))


def _is_background_name(label: str) -> bool:
    """True for any background cluster name ('BACKGROUND', 'BACKGROUND_2', ...)."""
    return label == "BACKGROUND" or label.startswith("BACKGROUND_")


class ColorAnalyzer:

    def __init__(self):
        self.color_space = "lab"

    def prepare_features(
        self, roi_bgr: np.ndarray, color_space: str = "lab"
    ) -> np.ndarray:
        if color_space == "lab":
            # Floating-point CIELAB (L in 0-100, a/b in -127..127). The 8-bit
            # encoding scales L by 255/100 but only offsets a/b by +128, which
            # stretches the lightness axis ~2.55x and skews both KMeans
            # distances and background detection. Float LAB keeps all three
            # axes in their true perceptual proportions.
            lab = cv2.cvtColor(roi_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
            return lab.reshape(-1, 3)
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

        # KMeans requires at least as many samples as clusters; tiny ROIs
        # must degrade gracefully instead of raising inside sklearn.
        effective_k = max(1, min(n_colors, features.shape[0]))

        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            random_state=42,
            max_iter=max_iter,
            n_init=n_init,
            batch_size=min(1024, len(features)),
        )
        labels = kmeans.fit_predict(features)

        return labels.reshape(roi_bgr.shape[:2]), kmeans

    @staticmethod
    def _assign_nearest_labels(
        features: np.ndarray, centers: np.ndarray, chunk_size: int = 262_144
    ) -> np.ndarray:
        """Assign every pixel to its nearest cluster center.

        Works on chunks so the (N, k, 3) difference tensor stays small even
        for very large ROIs. This is what allows clustering to be *fitted*
        on a downsampled image while coverage is still computed from every
        full-resolution pixel exactly.
        """
        centers = np.asarray(centers, dtype=np.float32)
        n = features.shape[0]
        labels = np.empty(n, dtype=np.int32)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            block = features[start:end]
            dists = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels[start:end] = np.argmin(dists, axis=1)
        return labels

    def get_representative_colors(
        self, kmeans, color_space: str = "lab"
    ) -> List[Tuple[int, int, int]]:
        # Work on a copy so the KMeans model's centers are not mutated.
        centers = kmeans.cluster_centers_.copy()

        if color_space == "lab":
            # Centers are floating-point CIELAB; decode through OpenCV's
            # float path (expects L in 0-100, a/b in -127..127) which
            # returns linear BGR in 0-1. Round instead of truncating to
            # keep representative colors accurate to within half a step.
            bgr01 = cv2.cvtColor(
                centers.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_Lab2BGR
            )
            bgr255 = np.clip(np.rint(bgr01 * 255.0), 0, 255).astype(np.uint8)
            return [tuple(int(v) for v in c[0]) for c in bgr255]
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
            centers_uint8 = np.rint(centers).astype(np.uint8).reshape(-1, 1, 3)
            bgr_centers = cv2.cvtColor(centers_uint8, cv2.COLOR_HSV2BGR)
            return [tuple(int(v) for v in c[0]) for c in bgr_centers]
        else:
            raise ValueError(f"Unsupported color space: {color_space}")

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

        # Fit KMeans on at most MAX_CLUSTER_PIXELS sampled pixels (speed),
        # then assign EVERY full-resolution pixel to its nearest center in
        # the same feature space. The previous approach upscaled the
        # downsampled label map with INTER_NEAREST, which quantized
        # coverage into blocks of ~(1/scale)^2 pixels: thin strokes, text
        # and halftone dots vanished or were miscounted. Exact per-pixel
        # assignment keeps coverage percentages accurate.
        small_img, _scale = self._downsample_image(roi_bgr)
        _, kmeans = self.cluster_colors(small_img, n_colors, color_space)

        features_full = self.prepare_features(roi_bgr, color_space)
        labels = self._assign_nearest_labels(
            features_full, kmeans.cluster_centers_
        ).reshape(h, w)

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

        bg_seen = 0
        for label_id, count in zip(unique_labels, counts):
            label_id = int(label_id)
            count = int(count)
            bgr = rep_colors[label_id]
            name = classify_color_name(bgr)

            is_background = False
            if bg_lab is not None:
                # Compare in true CIELAB (CIE76 Delta-E): the tolerance is
                # now in perceptually meaningful units where ~1.0 is one
                # just-noticeable difference, instead of the distorted
                # 8-bit Lab scale that amplified lightness ~2.55x.
                diff = _color_lab_distance(bgr, bg_lab)
                is_background = diff < background_tolerance

            if is_background:
                # Give every background cluster a unique name so masks are
                # never lost to dict-key collisions downstream.
                bg_seen += 1
                label = "BACKGROUND" if bg_seen == 1 else f"BACKGROUND_{bg_seen}"
            else:
                label = f"{name}_{label_id}"

            cluster = ColorCluster(
                label=label,
                representative_color_bgr=tuple(int(v) for v in bgr),
                representative_color_rgb=tuple(int(v) for v in (bgr[2], bgr[1], bgr[0])),
                pixel_count=count,
                coverage_percent=float((count / total_pixels) * 100),
                area_cm2=0.0,
                mask=self._create_mask(labels, label_id, (h, w)),
                visible=True,
            )
            clusters.append(cluster)

        bg_clusters = [c for c in clusters if _is_background_name(c.label)]
        non_bg_clusters = [c for c in clusters if not _is_background_name(c.label)]
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
            # All background variants ('BACKGROUND', 'BACKGROUND_2', ...)
            # deliberately share one merge key, so multiple paper-colored
            # clusters combine into a single exact-count BACKGROUND entry
            # instead of colliding in the per-color mask dictionary.
            if _is_background_name(c.label):
                base_name = "BACKGROUND"
            else:
                # Non-background labels are always "<color>_<cluster_id>",
                # so stripping the trailing id recovers the exact color
                # name. Splitting on the FIRST underscore would wrongly
                # collapse distinct colors such as YELLOW_GREEN into
                # YELLOW and corrupt their coverage percentages.
                base_name = (
                    c.label.rsplit("_", 1)[0] if "_" in c.label else c.label
                )

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
