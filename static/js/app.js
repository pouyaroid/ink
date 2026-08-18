/**
 * Ink Coverage Analyzer - Frontend Application (Persian)
 */

// ==================== State ====================
const state = {
    filename: null,
    sessionId: null,
    originalWidth: 0,
    originalHeight: 0,
    displayWidth: 0,
    displayHeight: 0,
    roi: null,
    roiOriginal: null,
    analysisResult: null,
    visibleColors: {},
};

// ==================== DOM Elements ====================
const els = {
    fileInput: document.getElementById('file-input'),
    uploadInfo: document.getElementById('upload-info'),
    imageInfo: document.getElementById('image-info'),
    infoFilename: document.getElementById('info-filename'),
    infoDims: document.getElementById('info-dims'),
    stepRoi: document.getElementById('step-roi'),
    stepParams: document.getElementById('step-params'),
    stepResults: document.getElementById('step-results'),
    canvas: document.getElementById('image-canvas'),
    roiPixels: document.getElementById('roi-pixels'),
    roiInfo: document.getElementById('roi-info'),
    numColors: document.getElementById('num-colors'),
    numColorsVal: document.getElementById('num-colors-val'),
    bgTolerance: document.getElementById('bg-tolerance'),
    bgToleranceVal: document.getElementById('bg-tolerance-val'),
    gaussianKsize: document.getElementById('gaussian-ksize'),
    gaussianKsizeVal: document.getElementById('gaussian-ksize-val'),
    btnAnalyze: document.getElementById('btn-analyze'),
    btnClearRoi: document.getElementById('btn-clear-roi'),
    loading: document.getElementById('loading'),
    roiPreview: document.getElementById('roi-preview'),
    segPreview: document.getElementById('seg-preview'),
    summary: document.getElementById('summary'),
    colorTableBody: document.getElementById('color-table-body'),
    masksGrid: document.getElementById('masks-grid'),
    btnStartNew: document.getElementById('btn-start-new'),
    btnResetSession: document.getElementById('btn-reset-session'),
};

// ==================== Canvas ROI Drawing ====================
let isDrawing = false;
let startX = 0;
let startY = 0;

const ctx = els.canvas.getContext('2d');
let loadedImage = null;

function initCanvasEvents() {
    els.canvas.addEventListener('mousedown', onMouseDown);
    els.canvas.addEventListener('mousemove', onMouseMove);
    els.canvas.addEventListener('mouseup', onMouseUp);
    els.canvas.addEventListener('mouseleave', onMouseUp);
}

function onMouseDown(e) {
    if (!loadedImage) return;
    isDrawing = true;
    const rect = els.canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
}

function onMouseMove(e) {
    if (!isDrawing || !loadedImage) return;
    const rect = els.canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);

    state.roi = { x, y, w, h };
    redrawCanvas();
}

function onMouseUp(e) {
    if (!isDrawing) return;
    isDrawing = false;

    if (state.roi && state.roi.w > 5 && state.roi.h > 5) {
        const scaleX = state.originalWidth / state.displayWidth;
        const scaleY = state.originalHeight / state.displayHeight;

        state.roiOriginal = {
            x: Math.round(state.roi.x * scaleX),
            y: Math.round(state.roi.y * scaleY),
            w: Math.round(state.roi.w * scaleX),
            h: Math.round(state.roi.h * scaleY),
        };

        const pixelCount = state.roiOriginal.w * state.roiOriginal.h;
        els.roiPixels.textContent = `${state.roiOriginal.w} × ${state.roiOriginal.h} پیکسل (${pixelCount.toLocaleString()} پیکسل)`;
        els.roiInfo.classList.remove('hidden');
        els.stepParams.classList.remove('hidden');
        els.btnAnalyze.disabled = false;
    } else {
        state.roi = null;
        state.roiOriginal = null;
        els.roiInfo.classList.add('hidden');
        els.stepParams.classList.add('hidden');
        els.btnAnalyze.disabled = true;
        redrawCanvas();
    }
}

function redrawCanvas() {
    if (!loadedImage) return;
    ctx.drawImage(loadedImage, 0, 0, els.canvas.width, els.canvas.height);

    if (state.roi) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, els.canvas.width, state.roi.y);
        ctx.fillRect(0, state.roi.y, state.roi.x, state.roi.h);
        ctx.fillRect(state.roi.x + state.roi.w, state.roi.y, els.canvas.width - state.roi.x - state.roi.w, state.roi.h);
        ctx.fillRect(0, state.roi.y + state.roi.h, els.canvas.width, els.canvas.height - state.roi.y - state.roi.h);

        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(state.roi.x, state.roi.y, state.roi.w, state.roi.h);
        ctx.setLineDash([]);

        if (state.roiOriginal) {
            ctx.fillStyle = '#00d4ff';
            ctx.font = '12px monospace';
            const label = `${state.roiOriginal.w} × ${state.roiOriginal.h} px`;
            ctx.fillText(label, state.roi.x + 4, state.roi.y - 6);
        }
    }
}

// ==================== Session Cleanup ====================
async function cleanupSession() {
    if (!state.sessionId) return;
    const formData = new FormData();
    formData.append('session_id', state.sessionId);
    state.sessionId = null;
    try {
        await fetch('/cleanup', { method: 'POST', body: formData });
    } catch (_) { /* best-effort */ }
}

window.addEventListener('beforeunload', () => {
    if (!state.sessionId) return;
    const payload = new FormData();
    payload.append('session_id', state.sessionId);
    navigator.sendBeacon('/cleanup', payload);
});

// ==================== File Upload ====================
els.fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    await cleanupSession();
    els.uploadInfo.textContent = 'در حال آپلود...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/upload', { method: 'POST', body: formData });
        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || 'آپلود ناموفق بود');
        }

        state.filename = data.filename;
        state.originalWidth = data.width;
        state.originalHeight = data.height;

        els.infoFilename.textContent = file.name;
        els.infoDims.textContent = `${data.width} × ${data.height} پیکسل`;
        els.imageInfo.classList.remove('hidden');
        els.uploadInfo.textContent = file.name;

        const img = new Image();
        img.onload = () => {
            loadedImage = img;

            const maxDisplayWidth = Math.min(1100, window.innerWidth - 80);
            const scale = Math.min(maxDisplayWidth / img.width, 1);
            state.displayWidth = Math.round(img.width * scale);
            state.displayHeight = Math.round(img.height * scale);

            els.canvas.width = state.displayWidth;
            els.canvas.height = state.displayHeight;

            redrawCanvas();
            els.stepRoi.classList.remove('hidden');
            els.stepResults.classList.add('hidden');
        };
        img.src = data.path;
    } catch (err) {
        els.uploadInfo.textContent = `خطا: ${err.message}`;
    }
});

// ==================== Clear ROI ====================
els.btnClearRoi.addEventListener('click', () => {
    state.roi = null;
    state.roiOriginal = null;
    els.roiInfo.classList.add('hidden');
    els.stepParams.classList.add('hidden');
    els.stepResults.classList.add('hidden');
    els.btnAnalyze.disabled = true;
    redrawCanvas();
});

// ==================== Slider Updates ====================
els.numColors.addEventListener('input', (e) => {
    els.numColorsVal.textContent = e.target.value;
});

els.bgTolerance.addEventListener('input', (e) => {
    els.bgToleranceVal.textContent = e.target.value;
});

els.gaussianKsize.addEventListener('input', (e) => {
    els.gaussianKsizeVal.textContent = e.target.value;
});

// ==================== Analysis ====================
els.btnAnalyze.addEventListener('click', async () => {
    if (!state.filename || !state.roiOriginal) return;

    els.loading.classList.remove('hidden');
    els.stepResults.classList.add('hidden');
    els.btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append('filename', state.filename);
    formData.append('roi_x', state.roiOriginal.x);
    formData.append('roi_y', state.roiOriginal.y);
    formData.append('roi_width', state.roiOriginal.w);
    formData.append('roi_height', state.roiOriginal.h);
    formData.append('length_mm', document.getElementById('length-mm').value);
    formData.append('width_mm', document.getElementById('width-mm').value);
    formData.append('num_colors', els.numColors.value);
    formData.append('color_space', document.getElementById('color-space').value);
    formData.append('apply_gaussian', document.getElementById('apply-gaussian').checked);
    formData.append('gaussian_ksize', els.gaussianKsize.value);
    formData.append('apply_median', document.getElementById('apply-median').checked);
    formData.append('median_ksize', 5);
    formData.append('background_tolerance', els.bgTolerance.value);

    if (document.getElementById('enable-bg').checked) {
        formData.append('enable_bg', 'true');
    }

    try {
        const resp = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.detail || 'تحلیل ناموفق بود');
        }

        await cleanupSession();
        state.sessionId = data.session_id;
        state.analysisResult = data;
        displayResults(data);
    } catch (err) {
        alert(`خطا در تحلیل: ${err.message}`);
    } finally {
        els.loading.classList.add('hidden');
        els.btnAnalyze.disabled = false;
    }
});

// ==================== Display Results ====================
function displayResults(data) {
    els.stepResults.classList.remove('hidden');

    drawRoiPreview();

    const segImg = new Image();
    segImg.onload = () => {
        const segCtx = els.segPreview.getContext('2d');
        els.segPreview.width = segImg.width;
        els.segPreview.height = segImg.height;
        segCtx.drawImage(segImg, 0, 0);
    };
    segImg.src = data.segmentation_base64;

    const s = data.summary;
    els.summary.innerHTML = `
        <span>تعداد پیکسل: <strong>${s.total_pixels.toLocaleString()}</strong></span>
        <span>ابعاد فیزیکی: <strong>${s.physical_length_mm} × ${s.physical_width_mm} میلی‌متر</strong></span>
        <span>مساحت کل: <strong>${s.total_area_mm2.toLocaleString()} میلی‌متر مربع</strong></span>
        <span>مساحت چاپ شده: <strong>${s.printed_area_mm2.toLocaleString()} میلی‌متر مربع (${s.printed_coverage_percent}٪)</strong></span>
    `;

    els.colorTableBody.innerHTML = '';
    state.visibleColors = {};

    data.colors.forEach((color, idx) => {
        state.visibleColors[color.name] = true;

        const row = document.createElement('tr');
        const rgb = color.representative_color_rgb;
        const colorHex = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

        row.innerHTML = `
            <td><strong>${color.name}</strong></td>
            <td><span class="color-preview" style="background:${colorHex}"></span></td>
            <td>${color.pixel_count.toLocaleString()}</td>
            <td>${color.coverage_percent}%</td>
            <td>${color.area_mm2.toLocaleString()} میلی‌متر مربع</td>
            <td>
                <button class="mask-btn active" data-color="${color.name}" data-action="toggle">نمایش</button>
                <button class="mask-btn" data-color="${color.name}" data-action="mask">ماسک</button>
            </td>
        `;

        row.querySelector('[data-action="toggle"]').addEventListener('click', (e) => {
            state.visibleColors[color.name] = !state.visibleColors[color.name];
            e.target.classList.toggle('active');
            e.target.textContent = state.visibleColors[color.name] ? 'نمایش' : 'پنهان';
        });

        row.querySelector('[data-action="mask"]').addEventListener('click', () => {
            showMask(color.name);
        });

        els.colorTableBody.appendChild(row);
    });

    drawMasksGrid(data);
}

// ==================== ROI Preview ====================
function drawRoiPreview() {
    if (!loadedImage || !state.roiOriginal) return;

    const roiCtx = els.roiPreview.getContext('2d');
    const roi = state.roiOriginal;

    els.roiPreview.width = roi.w;
    els.roiPreview.height = roi.h;
    roiCtx.drawImage(
        loadedImage,
        roi.x, roi.y, roi.w, roi.h,
        0, 0, roi.w, roi.h
    );
}

// ==================== Masks Grid ====================
function drawMasksGrid(data) {
    els.masksGrid.innerHTML = '';

    data.colors.forEach((color, idx) => {
        const item = document.createElement('div');
        item.className = 'mask-item';

        const rgb = color.representative_color_rgb;
        const colorHex = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

        item.innerHTML = `
            <h4 style="color:${colorHex}">${color.name}</h4>
            <canvas id="mask-canvas-${idx}" width="200" height="150"></canvas>
            <p>${color.pixel_count.toLocaleString()} پیکسل (${color.coverage_percent}٪)</p>
        `;
        els.masksGrid.appendChild(item);

        const canvas = document.getElementById(`mask-canvas-${idx}`);
        const maskCtx = canvas.getContext('2d');

        if (data.masks && data.masks[color.name]) {
            const maskImg = new Image();
            maskImg.onload = () => {
                canvas.width = maskImg.width;
                canvas.height = maskImg.height;

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskImg.width;
                tempCanvas.height = maskImg.height;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(maskImg, 0, 0);

                const maskData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
                const colored = maskCtx.createImageData(canvas.width, canvas.height);

                for (let i = 0; i < maskData.data.length; i += 4) {
                    if (maskData.data[i] > 128) {
                        colored.data[i] = rgb[0];
                        colored.data[i + 1] = rgb[1];
                        colored.data[i + 2] = rgb[2];
                        colored.data[i + 3] = 200;
                    } else {
                        colored.data[i] = 30;
                        colored.data[i + 1] = 30;
                        colored.data[i + 2] = 40;
                        colored.data[i + 3] = 255;
                    }
                }
                maskCtx.putImageData(colored, 0, 0);
            };
            maskImg.src = data.masks[color.name];
        }
    });
}

// ==================== Show Mask ====================
function showMask(colorName) {
    if (!state.analysisResult || !state.analysisResult.masks) return;
    const maskUrl = state.analysisResult.masks[colorName];
    if (maskUrl) {
        window.open(maskUrl, '_blank');
    }
}

// ==================== Start New ====================
els.btnStartNew.addEventListener('click', () => {
    state.filename = null;
    state.sessionId = null;
    state.originalWidth = 0;
    state.originalHeight = 0;
    state.displayWidth = 0;
    state.displayHeight = 0;
    state.roi = null;
    state.roiOriginal = null;
    state.analysisResult = null;
    state.visibleColors = {};

    loadedImage = null;

    els.canvas.width = 1;
    els.canvas.height = 1;
    ctx.clearRect(0, 0, 1, 1);

    els.fileInput.value = '';
    els.uploadInfo.textContent = 'فایلی انتخاب نشده';
    els.imageInfo.classList.add('hidden');
    els.stepRoi.classList.add('hidden');
    els.stepParams.classList.add('hidden');
    els.stepResults.classList.add('hidden');
    els.roiInfo.classList.add('hidden');
    els.loading.classList.add('hidden');
    els.btnAnalyze.disabled = true;

    document.getElementById('length-mm').value = 700;
    document.getElementById('width-mm').value = 500;
    els.numColors.value = 5;
    els.numColorsVal.textContent = '5';
    document.getElementById('color-space').value = 'lab';
    document.getElementById('enable-bg').checked = true;
    els.bgTolerance.value = 30;
    els.bgToleranceVal.textContent = '30';
    document.getElementById('apply-gaussian').checked = true;
    els.gaussianKsize.value = 5;
    els.gaussianKsizeVal.textContent = '5';
    document.getElementById('apply-median').checked = false;

    els.colorTableBody.innerHTML = '';
    els.masksGrid.innerHTML = '';
    els.summary.innerHTML = '';
});

// ==================== Reset Session ====================
els.btnResetSession.addEventListener('click', async () => {
    if (!state.sessionId) return;

    const formData = new FormData();
    formData.append('session_id', state.sessionId);
    if (state.filename) {
        formData.append('filename', state.filename);
    }

    state.sessionId = null;
    state.filename = null;
    state.analysisResult = null;

    els.colorTableBody.innerHTML = '';
    els.masksGrid.innerHTML = '';
    els.summary.innerHTML = '';
    els.stepResults.classList.add('hidden');

    try {
        await fetch('/reset-session', { method: 'POST', body: formData });
    } catch (_) { /* best-effort */ }
});

// ==================== Init ====================
initCanvasEvents();
