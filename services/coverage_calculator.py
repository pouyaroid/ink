"""
Coverage calculation service for converting pixel counts to physical areas.
"""

from typing import List, Dict
from .color_analyzer import ColorCluster


class CoverageCalculator:
    """Calculates coverage percentages and physical areas."""

    def calculate_areas(
        self,
        clusters: List[ColorCluster],
        total_pixels: int,
        length_mm: float,
        width_mm: float,
    ) -> List[Dict]:
        """
        Calculate coverage percentage and real area for each color cluster.

        Formulas:
        ----------
        coverage_percent = (pixel_count / total_pixels) × 100

        real_area_mm2 = length_mm × width_mm

        color_area_mm2 = real_area_mm2 × (pixel_count / total_pixels)
                       = real_area_mm2 × (coverage_percent / 100)

        Parameters:
        -----------
        clusters : list of ColorCluster
            Detected color clusters with pixel counts
        total_pixels : int
            Total number of pixels in the ROI
        length_mm : float
            Physical length of the ROI in millimeters
        width_mm : float
            Physical width of the ROI in millimeters

        Returns:
        --------
        list of dict with detailed results for each color
        """
        if total_pixels == 0:
            raise ValueError("Total pixels cannot be zero")

        if length_mm <= 0 or width_mm <= 0:
            raise ValueError("Physical dimensions must be positive")

        real_area_mm2 = length_mm * width_mm

        results = []
        for cluster in clusters:
            pixel_count = int(cluster.pixel_count)
            coverage_pct = (pixel_count / int(total_pixels)) * 100
            color_area_mm2 = real_area_mm2 * (pixel_count / int(total_pixels))
            pixels_per_mm2 = pixel_count / real_area_mm2 if real_area_mm2 > 0 else 0

            rgb = tuple(int(v) for v in cluster.representative_color_rgb)
            bgr = tuple(int(v) for v in cluster.representative_color_bgr)

            results.append(
                {
                    "name": str(cluster.label),
                    "pixel_count": pixel_count,
                    "coverage_percent": float(round(coverage_pct, 2)),
                    "area_mm2": float(round(color_area_mm2, 2)),
                    "pixels_per_mm2": float(round(pixels_per_mm2, 2)),
                    "representative_color_rgb": rgb,
                    "representative_color_bgr": bgr,
                    "visible": bool(cluster.visible),
                }
            )

        # Sort by coverage percent descending
        results.sort(key=lambda x: x["coverage_percent"], reverse=True)

        return results

    def get_summary(
        self,
        results: List[Dict],
        total_pixels: int,
        length_mm: float,
        width_mm: float,
    ) -> Dict:
        """Get summary statistics."""
        total_area_mm2 = length_mm * width_mm
        printed_area = sum(
            r["area_mm2"] for r in results if r["name"] != "BACKGROUND"
        )
        bg_area = sum(
            r["area_mm2"] for r in results if r["name"] == "BACKGROUND"
        )

        return {
            "total_pixels": int(total_pixels),
            "physical_length_mm": float(length_mm),
            "physical_width_mm": float(width_mm),
            "total_area_mm2": float(round(total_area_mm2, 2)),
            "printed_area_mm2": float(round(printed_area, 2)),
            "background_area_mm2": float(round(bg_area, 2)),
            "printed_coverage_percent": float(round(
                (printed_area / total_area_mm2) * 100 if total_area_mm2 > 0 else 0, 2
            )),
        }

    def validate_roi(
        self, x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> bool:
        """Validate ROI coordinates are within image bounds."""
        if x < 0 or y < 0:
            return False
        if x + w > img_w or y + h > img_h:
            return False
        if w <= 0 or h <= 0:
            return False
        return True
