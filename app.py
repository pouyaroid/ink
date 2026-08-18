"""
FastAPI application for ink coverage analysis prototype.
"""

import os
import shutil
import uuid
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, FileResponse
from fastapi.requests import Request
from typing import Optional
import base64

from services.image_processor import ImageProcessor
from services.color_analyzer import ColorAnalyzer
from services.coverage_calculator import CoverageCalculator


UPLOAD_DIR = Path("uploads")
SESSIONS_DIR = Path("sessions")
SESSION_EXPIRY_HOURS = 2


def _cleanup_expired_sessions():
    """Remove session directories older than SESSION_EXPIRY_HOURS."""
    now = datetime.now(timezone.utc)
    for session_dir in SESSIONS_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        mtime = datetime.fromtimestamp(session_dir.stat().st_mtime, tz=timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600
        if age_hours >= SESSION_EXPIRY_HOURS:
            shutil.rmtree(session_dir, ignore_errors=True)


def _cleanup_expired_uploads():
    """Remove uploaded files older than SESSION_EXPIRY_HOURS."""
    now = datetime.now(timezone.utc)
    for filepath in UPLOAD_DIR.iterdir():
        if not filepath.is_file():
            continue
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
        age_hours = (now - mtime).total_seconds() / 3600
        if age_hours >= SESSION_EXPIRY_HOURS:
            filepath.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)
    _cleanup_expired_sessions()
    _cleanup_expired_uploads()
    yield


UPLOAD_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Ink Coverage Analyzer", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/sessions", StaticFiles(directory="sessions"), name="sessions")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

processor = ImageProcessor()
analyzer = ColorAnalyzer()
calculator = CoverageCalculator()


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    _cleanup_expired_uploads()

    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / filename

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    image = processor.load_image(str(filepath))
    dims = processor.get_image_dimensions(image)

    return {
        "filename": filename,
        "path": f"/uploads/{filename}",
        "width": int(dims["width"]),
        "height": int(dims["height"]),
    }


def _get_dominant_color(image: np.ndarray) -> tuple:
    """Find the most common color in the image by simple binning."""
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    # Quantize to reduce unique colors
    quantized = (pixels // 16) * 16
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = unique[np.argmax(counts)]
    return (int(dominant[0]), int(dominant[1]), int(dominant[2]))


@app.post("/analyze")
def analyze_image(
    filename: str = Form(...),
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_width: int = Form(...),
    roi_height: int = Form(...),
    length_mm: float = Form(...),
    width_mm: float = Form(...),
    num_colors: int = Form(5),
    color_space: str = Form("lab"),
    background_r: Optional[int] = Form(None),
    background_g: Optional[int] = Form(None),
    background_b: Optional[int] = Form(None),
    background_tolerance: int = Form(30),
    apply_gaussian: bool = Form(True),
    gaussian_ksize: int = Form(5),
    apply_median: bool = Form(False),
    median_ksize: int = Form(5),
    enable_bg: bool = Form(False),
):
    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    session_id = uuid.uuid4().hex
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    _cleanup_expired_sessions()

    image = processor.load_image(str(filepath))
    img_h, img_w = image.shape[:2]

    if not calculator.validate_roi(roi_x, roi_y, roi_width, roi_height, img_w, img_h):
        raise HTTPException(status_code=400, detail="Invalid ROI coordinates")

    roi = processor.extract_roi(image, roi_x, roi_y, roi_width, roi_height)

    roi_processed = processor.preprocess(
        roi,
        apply_gaussian=apply_gaussian,
        apply_median=apply_median,
        gaussian_ksize=gaussian_ksize,
        median_ksize=median_ksize,
    )

    num_colors = max(2, min(12, num_colors))

    bg_color = None
    if background_r is not None and background_g is not None and background_b is not None:
        bg_color = (background_b, background_g, background_r)
    elif enable_bg:
        bg_color = _get_dominant_color(roi_processed)

    analysis = analyzer.analyze_colors(
        roi_processed,
        n_colors=num_colors,
        color_space=color_space,
        background_color=bg_color,
        background_tolerance=background_tolerance,
    )

    results = calculator.calculate_areas(
        analysis["clusters"],
        analysis["total_pixels"],
        length_mm,
        width_mm,
    )

    summary = calculator.get_summary(results, analysis["total_pixels"], length_mm, width_mm)

    seg_filename = f"seg_{filename}"
    seg_path = session_dir / seg_filename
    cv2.imwrite(str(seg_path), analysis["segmentation_preview"])

    masks = {}
    for i, cluster in enumerate(analysis["clusters"]):
        mask_filename = f"mask_{filename}_{i}.png"
        mask_path = session_dir / mask_filename
        cv2.imwrite(str(mask_path), cluster.mask)
        masks[cluster.label] = f"/sessions/{session_id}/{mask_filename}"

    _, seg_encoded = cv2.imencode(".jpg", analysis["segmentation_preview"])
    seg_base64 = base64.b64encode(seg_encoded).decode("utf-8")

    def to_native(val):
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    response = {
        "session_id": session_id,
        "roi": {
            "x": int(roi_x),
            "y": int(roi_y),
            "width": int(roi_width),
            "height": int(roi_height),
        },
        "physical_dimensions": {
            "length_mm": float(length_mm),
            "width_mm": float(width_mm),
            "area_mm2": round(float(length_mm * width_mm), 2),
        },
        "summary": {k: to_native(v) for k, v in summary.items()},
        "colors": [{k: to_native(v) for k, v in color.items()} for color in results],
        "segmentation_preview": f"/sessions/{session_id}/{seg_filename}",
        "segmentation_base64": f"data:image/jpeg;base64,{seg_base64}",
        "masks": masks,
    }

    return JSONResponse(content=response)


@app.post("/cleanup")
async def cleanup_session(session_id: str = Form(...)):
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists() and session_dir.is_dir():
        shutil.rmtree(session_dir, ignore_errors=True)
    return {"status": "ok"}


@app.post("/reset-session")
async def reset_session(
    session_id: str = Form(...),
    filename: str = Form(""),
):
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists() and session_dir.is_dir():
        shutil.rmtree(session_dir, ignore_errors=True)

    if filename:
        upload_path = UPLOAD_DIR / filename
        if upload_path.exists():
            upload_path.unlink(missing_ok=True)

    return {"status": "ok"}


@app.post("/get-mask")
async def get_mask(
    session_id: str = Form(...),
    filename: str = Form(...),
    color_name: str = Form(...),
):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    for mask_file in session_dir.glob(f"mask_{filename}_*.png"):
        if color_name in mask_file.stem:
            return FileResponse(mask_file, media_type="image/png")

    raise HTTPException(status_code=404, detail="Mask not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
