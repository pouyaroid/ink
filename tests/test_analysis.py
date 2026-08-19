"""
Test script for the ink coverage analysis pipeline.
Generates test images programmatically and validates detection accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from services.image_processor import ImageProcessor
from services.color_analyzer import ColorAnalyzer
from services.coverage_calculator import CoverageCalculator

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

processor = ImageProcessor()
analyzer = ColorAnalyzer()
calculator = CoverageCalculator()


def create_test_image_solid(width=800, height=600, color_bgr=(255, 0, 0)):
    """Create a solid color image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def create_test_image_two_colors(width=800, height=600):
    """Create an image with 50% black, 50% white."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :width // 2] = (0, 0, 0)  # Black
    img[:, width // 2:] = (255, 255, 255)  # White
    return img


def create_test_image_multi_colors(width=800, height=600):
    """Create an image with 4 color blocks (25% each)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    hw, hh = width // 2, height // 2
    img[:hh, :hw] = (255, 0, 0)    # Blue (BGR)
    img[:hh, hw:] = (0, 255, 0)    # Green
    img[hh:, :hw] = (0, 0, 255)    # Red
    img[hh:, hw:] = (0, 255, 255)  # Yellow
    return img


def create_test_image_with_antialiasing(width=800, height=600):
    """Create image with anti-aliased edges."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw a circle with anti-aliasing
    cv2.circle(img, (width // 2, height // 2), 200, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # The rest is black
    return img


def create_test_image_jpeg_compressed(width=800, height=600):
    """Create JPEG-compressed image to test noise tolerance."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :width // 2] = (0, 0, 200)  # Red-ish
    img[:, width // 2:] = (200, 0, 0)  # Blue-ish

    # Save as JPEG with low quality to introduce compression artifacts
    path = OUTPUT_DIR / "test_jpeg_input.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.imread(str(path))


def create_test_image_with_background(width=1000, height=800):
    """Create carton-like image with brown kraft background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Brown kraft cardboard background (BGR)
    img[:] = (40, 80, 120)

    # Carton area (white with printed colors)
    margin = 100
    cv2.rectangle(img, (margin, margin), (width - margin, height - margin), (255, 255, 255), -1)

    # Printed colors on the carton
    cv2.rectangle(img, (150, 150), (350, 350), (255, 0, 0), -1)    # Blue
    cv2.rectangle(img, (400, 150), (600, 350), (0, 0, 255), -1)    # Red
    cv2.rectangle(img, (150, 400), (350, 600), (0, 200, 200), -1)  # Yellow
    cv2.putText(img, "PRODUCT", (420, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    return img


def create_test_image_small_details(width=800, height=600):
    """Create image with small printed details."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # Light gray background

    # Small text and details
    for i in range(0, width, 40):
        cv2.line(img, (i, 0), (i, height), (100, 100, 100), 1)

    for j in range(0, height, 40):
        cv2.line(img, (0, j), (width, j), (100, 100, 100), 1)

    # Small colored dots
    for i in range(10):
        x = 100 + i * 60
        cv2.circle(img, (x, 300), 5, (0, 0, 255), -1)

    return img


def run_test(name, image, expected_colors, expected_coverage_range, width_cm=70, height_cm=50):
    """Run a single test case."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")

    # Save test image
    img_path = OUTPUT_DIR / f"test_{name.replace(' ', '_').lower()}.png"
    cv2.imwrite(str(img_path), image)
    print(f"  Image saved: {img_path}")

    h, w = image.shape[:2]
    print(f"  Image size: {w} × {h} = {w * h:,} pixels")

    # Use full image as ROI (for testing)
    roi = image.copy()

    # Analyze with default 5 colors
    n_colors = max(2, min(12, expected_colors if expected_colors else 5))
    result = analyzer.analyze_colors(roi, n_colors=n_colors, color_space="lab")

    # Calculate areas
    total_pixels = result["total_pixels"]
    areas = calculator.calculate_areas(
        result["clusters"], total_pixels, width_cm, height_cm
    )

    total_area_cm2 = width_cm * height_cm

    print(f"\n  ROI: Full image ({w} × {h})")
    print(f"  Physical: {width_cm} × {height_cm} cm = {total_area_cm2} cm²")
    print(f"  Detected {len(areas)} color groups:")
    print(f"  {'Color':<20} {'Pixels':>10} {'Coverage':>10} {'Area mm²':>10}")
    print(f"  {'-' * 55}")

    for c in areas:
        print(f"  {c['name']:<20} {c['pixel_count']:>10,} {c['coverage_percent']:>9}% {c['area_mm2']:>10,.2f}")

    # Validate
    all_coverage = sum(c["coverage_percent"] for c in areas)
    print(f"\n  Total coverage: {all_coverage:.2f}% (should be ~100%)")

    passed = True
    if abs(all_coverage - 100.0) > 1.0:
        print(f"  WARNING: Total coverage deviation > 1%")
        passed = False

    if expected_coverage_range:
        for expected_name, (low, high) in expected_coverage_range.items():
            found = [c for c in areas if expected_name.lower() in c["name"].lower()]
            if found:
                actual = found[0]["coverage_percent"]
                if low <= actual <= high:
                    print(f"  PASS: {expected_name} coverage {actual}% in range [{low}%, {high}%]")
                else:
                    print(f"  FAIL: {expected_name} coverage {actual}% NOT in range [{low}%, {high}%]")
                    passed = False
            else:
                print(f"  WARNING: Expected color '{expected_name}' not found")

    # Save segmentation preview
    seg_path = OUTPUT_DIR / f"seg_{name.replace(' ', '_').lower()}.png"
    cv2.imwrite(str(seg_path), result["segmentation_preview"])
    print(f"  Segmentation saved: {seg_path}")

    print(f"\n  Result: {'PASS' if passed else 'NEEDS REVIEW'}")
    return passed


def main():
    print("Ink Coverage Analyzer - Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Solid blue
    img = create_test_image_solid(color_bgr=(255, 0, 0))
    results["solid_blue"] = run_test(
        "Solid Blue", img, 2,
        {"BLUE": (90, 100)}
    )

    # Test 2: Two colors (50/50 black/white)
    img = create_test_image_two_colors()
    results["two_colors"] = run_test(
        "Two Colors", img, 3,
        {"BLACK": (45, 55), "WHITE": (45, 55)}
    )

    # Test 3: Multiple colors
    img = create_test_image_multi_colors()
    results["multi_colors"] = run_test(
        "Multiple Colors", img, 5,
        {"BLUE": (20, 30), "GREEN": (20, 30), "RED": (20, 30), "YELLOW": (20, 30)}
    )

    # Test 4: Anti-aliased edges
    img = create_test_image_with_antialiasing()
    results["anti_aliasing"] = run_test(
        "Anti-aliased Edges", img, 3,
        {"RED": (15, 25), "BLACK": (50, 65)}
    )

    # Test 5: JPEG compressed
    img = create_test_image_jpeg_compressed()
    results["jpeg_compressed"] = run_test(
        "JPEG Compressed", img, 3,
        {"RED": (40, 60), "BLUE": (40, 60)}
    )

    # Test 6: Background surrounding carton
    img = create_test_image_with_background()
    results["background_carton"] = run_test(
        "Background Carton", img, 6,
        {"BLUE": (8, 18), "RED": (8, 18), "YELLOW": (8, 18)}
    )

    # Test 7: Small details
    img = create_test_image_small_details()
    results["small_details"] = run_test(
        "Small Details", img, 4,
        {"GRAY": (60, 80), "WHITE": (15, 30)}
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"  Total: {total}")
    print(f"  Passed: {passed}")
    print(f"  Review: {total - passed}")
    print(f"\n  All test images saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
