"""
Tests for the color-analysis fixes:

1. HSV Hue round-trip (representative colors must map back to the real color).
2. LAB-based background removal (perceptual distance instead of BGR).

Runs standalone (`python tests/test_color_analysis.py`) or under pytest.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from services.color_analyzer import (
    ColorAnalyzer,
    classify_color_name,
    _bgr_to_lab,
    _color_lab_distance,
)
from services.coverage_calculator import CoverageCalculator

analyzer = ColorAnalyzer()
calculator = CoverageCalculator()

# BGR colors to exercise: blue, green, red, yellow, black, white
TEST_COLORS = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255),
}


def make_solid(color_bgr, w=64, h=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color_bgr
    return img


def top_cluster(res):
    """Return the largest non-background cluster."""
    fg = [c for c in res["clusters"] if c.label != "BACKGROUND"]
    return max(fg or res["clusters"], key=lambda c: c.pixel_count)


# ----------------------- HSV Hue round-trip -----------------------


def test_hsv_blue_not_shifted_to_green():
    """Regression: blue (H~120) must not come back as green (H~60).

    Blue in BGR is (255, 0, 0).  After HSV clustering the representative
    color must still be classified as BLUE, not shifted toward green which
    would happen if the Hue halving is not reversed.
    """
    cluster = top_cluster(analyzer.analyze_colors(make_solid((255, 0, 0)), n_colors=3, color_space="hsv"))
    b, g, r = cluster.representative_color_bgr
    # Blue has a high B channel and low R/G channels.
    assert b > 150, f"blue lost its B component: {(b, g, r)}"
    assert r < 120 and g < 120, f"blue shifted toward green or red: {(b, g, r)}"
    assert classify_color_name((b, g, r)) == "BLUE", f"got {classify_color_name((b, g, r))}"


def test_hsv_representative_color_recovery():
    """All solid colors must round-trip through HSV clustering unchanged."""
    for name, bgr in TEST_COLORS.items():
        cluster = top_cluster(analyzer.analyze_colors(make_solid(bgr), n_colors=3, color_space="hsv"))
        rec = cluster.representative_color_bgr
        dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(rec, bgr))))
        assert dist < 40, f"{name}: recovered {rec} too far from {bgr} (dist={dist:.1f})"
        assert classify_color_name(rec) == name, f"{name}: got {classify_color_name(rec)}"


def test_lab_representative_color_recovery():
    """CIELAB mode must be unaffected by the HSV fix (regression)."""
    for name, bgr in TEST_COLORS.items():
        cluster = top_cluster(analyzer.analyze_colors(make_solid(bgr), n_colors=3, color_space="lab"))
        rec = cluster.representative_color_bgr
        dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(rec, bgr))))
        assert dist < 40, f"{name}: recovered {rec} too far from {bgr} (dist={dist:.1f})"
        assert classify_color_name(rec) == name, f"{name}: got {classify_color_name(rec)}"


# ----------------------- LAB background removal -----------------------


def test_lab_distance_reflects_perception():
    """Similar colors -> small LAB distance; different colors -> large distance."""
    gray = (240, 240, 240)
    assert _color_lab_distance(gray, _bgr_to_lab(gray)) == 0.0
    assert _color_lab_distance((235, 235, 235), _bgr_to_lab((245, 245, 245))) < 15.0
    assert _color_lab_distance((0, 0, 255), _bgr_to_lab(gray)) > 50.0


def _two_tone(bg_bgr, fg_bgr, size=100, sq=20):
    """Create a two-tone image: solid background with a centered foreground square.

    Default sq=20 gives a 60×60 foreground (3600 px) in a 100×100 image.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = bg_bgr
    img[sq:size - sq, sq:size - sq] = fg_bgr
    return img


def test_background_removal_light_background():
    """Light gray background + red print: background flagged, red kept."""
    res = analyzer.analyze_colors(
        _two_tone((240, 240, 240), (0, 0, 255)),
        n_colors=4, color_space="lab",
        background_color=(240, 240, 240), background_tolerance=30,
    )
    labels = [c.label for c in res["clusters"]]
    assert "BACKGROUND" in labels, f"background not detected: {labels}"

    fg = [c for c in res["clusters"] if c.label != "BACKGROUND"]
    assert fg, "printed color was removed entirely"
    top = max(fg, key=lambda c: c.pixel_count)
    b, g, r = top.representative_color_bgr
    assert r > 150 and b < 120, f"foreground not red: {(b, g, r)}"
    # red square is 60x60 of 100x100 = 3600 px; allow some margin for
    # anti-aliasing / merging at cluster boundaries
    assert top.pixel_count > 2500, f"red region too small: {top.pixel_count}"


def test_background_removal_dark_background():
    """Dark background + white print: background flagged, white kept."""
    res = analyzer.analyze_colors(
        _two_tone((20, 20, 20), (255, 255, 255)),
        n_colors=4, color_space="lab",
        background_color=(20, 20, 20), background_tolerance=30,
    )
    labels = [c.label for c in res["clusters"]]
    assert "BACKGROUND" in labels, f"background not detected: {labels}"

    fg = [c for c in res["clusters"] if c.label != "BACKGROUND"]
    assert fg, "printed color was removed entirely"
    top = max(fg, key=lambda c: c.pixel_count)
    b, g, r = top.representative_color_bgr
    assert r > 200 and g > 200 and b > 200, f"foreground not white: {(b, g, r)}"


def test_background_removal_colored_background():
    """Brown kraft background + blue print: background flagged, blue kept."""
    brown = (40, 80, 120)  # brown kraft cardboard in BGR
    blue = (255, 0, 0)
    res = analyzer.analyze_colors(
        _two_tone(brown, blue),
        n_colors=4, color_space="lab",
        background_color=brown, background_tolerance=30,
    )
    labels = [c.label for c in res["clusters"]]
    assert "BACKGROUND" in labels, f"background not detected: {labels}"

    fg = [c for c in res["clusters"] if c.label != "BACKGROUND"]
    assert fg, "printed blue was removed entirely"
    top = max(fg, key=lambda c: c.pixel_count)
    b, g, r = top.representative_color_bgr
    assert b > 150, f"foreground not blue: {(b, g, r)}"


def test_background_removal_tolerance_sensitivity():
    """Tight tolerance keeps more clusters; loose tolerance merges them."""
    img = _two_tone((240, 240, 240), (0, 0, 255))

    tight = analyzer.analyze_colors(
        img, n_colors=4, color_space="lab",
        background_color=(240, 240, 240), background_tolerance=5,
    )
    loose = analyzer.analyze_colors(
        img, n_colors=4, color_space="lab",
        background_color=(240, 240, 240), background_tolerance=80,
    )

    tight_fg = [c for c in tight["clusters"] if c.label != "BACKGROUND"]
    loose_fg = [c for c in loose["clusters"] if c.label != "BACKGROUND"]

    # With very tight tolerance, background may not be detected
    # (depending on how close a cluster center is to exact background)
    # With loose tolerance, almost everything is labeled background
    assert len(loose_fg) <= len(tight_fg), (
        f"loose tolerance should have fewer foreground clusters: "
        f"loose={len(loose_fg)}, tight={len(tight_fg)}"
    )


def test_background_removal_with_hsv_clustering():
    """Background removal works regardless of clustering color space."""
    res_lab = analyzer.analyze_colors(
        _two_tone((240, 240, 240), (0, 0, 255)),
        n_colors=4, color_space="lab",
        background_color=(240, 240, 240), background_tolerance=30,
    )
    res_hsv = analyzer.analyze_colors(
        _two_tone((240, 240, 240), (0, 0, 255)),
        n_colors=4, color_space="hsv",
        background_color=(240, 240, 240), background_tolerance=30,
    )
    # Both should detect a BACKGROUND cluster
    assert "BACKGROUND" in [c.label for c in res_lab["clusters"]], "LAB mode: bg not detected"
    assert "BACKGROUND" in [c.label for c in res_hsv["clusters"]], "HSV mode: bg not detected"

    # Both should keep the red foreground
    for mode, res in [("lab", res_lab), ("hsv", res_hsv)]:
        fg = [c for c in res["clusters"] if c.label != "BACKGROUND"]
        assert fg, f"{mode}: printed color was removed entirely"
        top = max(fg, key=lambda c: c.pixel_count)
        b, g, r = top.representative_color_bgr
        assert r > 150, f"{mode}: foreground not red: {(b, g, r)}"


def test_lab_distance_color_pairs():
    """Verify LAB distances match perceptual similarity for common pairs."""
    # Same color → distance 0
    assert _color_lab_distance((128, 128, 128), _bgr_to_lab((128, 128, 128))) == 0.0

    # Similar grays → small distance
    d_gray = _color_lab_distance((200, 200, 200), _bgr_to_lab((210, 210, 210)))
    assert d_gray < 10.0, f"similar grays too far apart: {d_gray}"

    # Blue vs red → large distance (very different hues)
    d_blue_red = _color_lab_distance((255, 0, 0), _bgr_to_lab((0, 0, 255)))
    assert d_blue_red > 80.0, f"blue vs red too close: {d_blue_red}"

    # Blue vs green → large distance
    d_blue_green = _color_lab_distance((255, 0, 0), _bgr_to_lab((0, 255, 0)))
    assert d_blue_green > 60.0, f"blue vs green too close: {d_blue_green}"

    # Light gray vs white → moderate distance (both achromatic)
    d_gray_white = _color_lab_distance((200, 200, 200), _bgr_to_lab((255, 255, 255)))
    assert 10.0 < d_gray_white < 50.0, f"gray vs white unexpected: {d_gray_white}"


def test_background_not_labeled_when_disabled():
    """Without a background color, nothing is labeled BACKGROUND."""
    res = analyzer.analyze_colors(
        _two_tone((240, 240, 240), (0, 0, 255)),
        n_colors=4, color_space="lab",
    )
    assert not any(c.label == "BACKGROUND" for c in res["clusters"])


# ----------------------- Coverage / areas -----------------------


def test_coverage_percentages_accurate():
    """50/50 blue/black image: ~50% each, areas correct, sum ~100%."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = (255, 0, 0)
    img[:, 50:] = (0, 0, 0)

    res = analyzer.analyze_colors(img, n_colors=3, color_space="lab")
    areas = calculator.calculate_areas(res["clusters"], res["total_pixels"], 70.0, 50.0)

    total = sum(c["coverage_percent"] for c in areas)
    assert abs(total - 100.0) < 0.5, f"coverage sum {total:.2f} != 100"

    blue = [c for c in areas if "BLUE" in c["name"]]
    assert blue, "blue cluster missing"
    assert abs(blue[0]["coverage_percent"] - 50.0) < 2.0, f"blue {blue[0]['coverage_percent']:.2f}%"
    assert abs(blue[0]["area_mm2"] - 1750.0) < 70.0, f"blue area {blue[0]['area_mm2']:.2f}"


def _run_all():
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())