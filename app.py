import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from detector import detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app = FastAPI(
    title="Deepfake Detection - FaceForensics++ Lab",
    description="Interactive Web Studio for Deepfake Video & Image Detection with Grad-CAM Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Research assets mapping
RESEARCH_ASSETS = {
    "model_comparison": "model_comparison.png",
    "ensemble_confusion_matrix": "ensemble_confusion_matrix.png",
    "xception_confusion_matrix": "xception_confusion_matrix.png",
    "efficientnet_confusion_matrix": "efficientnet_confusion_matrix.png",
    "svm_confusion_matrix": "svm_confusion_matrix.png",
    "rf_confusion_matrix": "rf_confusion_matrix.png",
    "ensemble_sensitivity": "ensemble_sensitivity.png",
    "xception_training_curves": "xception_training_curves.png",
    "gradcam_heatmaps": "gradcam_heatmaps.png",
    "gradcam_failed_cases": "gradcam_failed_cases.png",
    "gradcam_celebdf_failed": "gradcam_celebdf_failed.png",
    "face_extraction_sample": "face_extraction_sample.png",
    "background_masking_sample": "background_masking_sample.png",
    "sample_real_video": "gradcam_real_output.mp4",
    "sample_fake_video": "gradcam_fake_output.mp4"
}

@app.get("/")
def read_root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "API active. Frontend index.html loading."})

@app.get("/assets/{asset_name}")
def get_research_asset(asset_name: str):
    filename = RESEARCH_ASSETS.get(asset_name, asset_name)
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        media_type = "video/mp4" if filename.endswith(".mp4") else "image/png"
        return FileResponse(file_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Asset not found")

@app.get("/api/health")
def get_health():
    return {
        "status": "online",
        "detector_ready": detector.is_ready,
        "efficientnet_loaded": detector.eff_model is not None,
        "xception_loaded": detector.xception_model is not None,
        "gradcam_ready": detector.grad_model is not None
    }

@app.get("/api/samples")
def list_samples():
    samples = [
        {
            "id": "sample_fake_video",
            "name": "Manipulated Deepfake Sequence (c23)",
            "type": "video",
            "file": "gradcam_fake_output.mp4",
            "description": "Deepfake synthesis video with facial landmark warping artifacts.",
            "url": "/assets/sample_fake_video",
            "expected": "FAKE"
        },
        {
            "id": "sample_real_video",
            "name": "Original Authentic Sequence (c23)",
            "type": "video",
            "file": "gradcam_real_output.mp4",
            "description": "Authentic YouTube interview sequence without facial manipulation.",
            "url": "/assets/sample_real_video",
            "expected": "REAL"
        },
        {
            "id": "face_extraction_sample",
            "name": "MTCNN Face Extraction Sample",
            "type": "image",
            "file": "face_extraction_sample.png",
            "description": "Aligned facial crops used for multi-model feature extraction.",
            "url": "/assets/face_extraction_sample",
            "expected": "IMAGE_BATCH"
        },
        {
            "id": "background_masking_sample",
            "name": "Elliptical Masked Face Sample",
            "type": "image",
            "file": "background_masking_sample.png",
            "description": "Boundary-isolated facial region to suppress background bias.",
            "url": "/assets/background_masking_sample",
            "expected": "MASKED"
        }
    ]
    return {"samples": samples}

@app.get("/api/metrics")
def get_metrics():
    return {
        "project": {
            "title": "Deepfake Detection - FaceForensics++",
            "institution": "Long Island University, Brooklyn",
            "course": "AI688-001 Image and Vision Computing (Spring 2026)",
            "professor": "Reda Nacif Elalaoui",
            "group": "Group 2: Vishal Patel & Saunil Patel",
            "dataset": "FaceForensics++ (c23 compression)",
            "cross_dataset": "Celeb-DF v2 (Higher quality fakes)"
        },
        "models": [
            {
                "name": "FFT + SVM",
                "type": "Frequency Domain Baseline",
                "accuracy": 75.33,
                "auc": 0.8124,
                "description": "Fast Fourier Transform spatial frequency features classified by Support Vector Machine."
            },
            {
                "name": "dlib + Random Forest",
                "type": "Geometric Feature Baseline",
                "accuracy": 76.74,
                "auc": 0.8694,
                "description": "68-point facial landmark distances analyzed via Random Forest ensemble."
            },
            {
                "name": "Xception",
                "type": "Deep Learning (Two-Stage Fine-tuning)",
                "accuracy": 92.67,
                "auc": 0.9881,
                "description": "Deep depthwise separable CNN with two-stage fine-tuning schedule."
            },
            {
                "name": "EfficientNet-B0 + SAM",
                "type": "Deep Learning (Spatial Attention)",
                "accuracy": 89.33,
                "auc": 0.9611,
                "description": "EfficientNet-B0 with custom Spatial Attention Module focusing on subtle warping boundaries."
            },
            {
                "name": "Equal Ensemble (0.5/0.5)",
                "type": "Optimized Ensemble",
                "accuracy": 95.33,
                "auc": 0.9767,
                "description": "Optimal weighted ensemble of Xception and EfficientNet determined through sensitivity analysis."
            },
            {
                "name": "Celeb-DF Cross-Dataset",
                "type": "Generalization Benchmark",
                "accuracy": 75.00,
                "auc": 0.7800,
                "description": "Zero-shot cross-dataset generalization test on Celeb-DF v2 higher-tier fakes (threshold 0.35)."
            }
        ],
        "ensemble_weights": [
            {"weights": "0.3 Xcep / 0.7 Eff", "accuracy": 91.33, "auc": 0.965},
            {"weights": "0.4 Xcep / 0.6 Eff", "accuracy": 93.33, "auc": 0.971},
            {"weights": "0.5 Xcep / 0.5 Eff (Optimal)", "accuracy": 95.33, "auc": 0.977},
            {"weights": "0.6 Xcep / 0.4 Eff", "accuracy": 94.67, "auc": 0.978},
            {"weights": "0.7 Xcep / 0.3 Eff", "accuracy": 93.33, "auc": 0.981},
            {"weights": "0.8 Xcep / 0.2 Eff", "accuracy": 93.00, "auc": 0.983}
        ],
        "confusion_matrices": {
            "ensemble": "/assets/ensemble_confusion_matrix",
            "xception": "/assets/xception_confusion_matrix",
            "efficientnet": "/assets/efficientnet_confusion_matrix",
            "svm": "/assets/svm_confusion_matrix",
            "rf": "/assets/rf_confusion_matrix"
        },
        "explainability_assets": {
            "gradcam_heatmaps": "/assets/gradcam_heatmaps",
            "gradcam_failed_cases": "/assets/gradcam_failed_cases",
            "gradcam_celebdf_failed": "/assets/gradcam_celebdf_failed",
            "ensemble_sensitivity": "/assets/ensemble_sensitivity",
            "xception_training_curves": "/assets/xception_training_curves"
        }
    }

@app.post("/api/predict")
async def predict_media(
    file: UploadFile = File(...),
    num_frames: int = Form(10),
    apply_mask: bool = Form(False),
    generate_heatmap: bool = Form(True),
    threshold: float = Form(0.50)
):
    if not detector.is_ready:
        raise HTTPException(status_code=503, detail="Detector model is not ready")

    filename = file.filename.lower()
    is_video = any(filename.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"])
    is_image = any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"])

    if not is_video and not is_image:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload an image or video file.")

    temp_path = None
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TEMP_DIR) as tmp:
            temp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        if is_video:
            result = detector.predict_video(
                temp_path,
                num_frames=max(3, min(num_frames, 30)),
                apply_mask=apply_mask,
                generate_heatmap=generate_heatmap,
                threshold=threshold
            )
        else:
            with open(temp_path, "rb") as f:
                img_bytes = f.read()
            result = detector.predict_image(
                img_bytes,
                apply_mask=apply_mask,
                generate_heatmap=generate_heatmap,
                threshold=threshold
            )

        result["filename"] = file.filename
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/api/predict-sample")
async def predict_sample(
    sample_id: str = Form(...),
    num_frames: int = Form(10),
    apply_mask: bool = Form(False),
    generate_heatmap: bool = Form(True),
    threshold: float = Form(0.50)
):
    filename = RESEARCH_ASSETS.get(sample_id)
    if not filename:
        raise HTTPException(status_code=404, detail="Unknown sample ID")

    sample_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample media file missing on server")

    is_video = sample_path.lower().endswith(".mp4")

    if is_video:
        result = detector.predict_video(
            sample_path,
            num_frames=max(3, min(num_frames, 30)),
            apply_mask=apply_mask,
            generate_heatmap=generate_heatmap,
            threshold=threshold
        )
    else:
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
        result = detector.predict_image(
            img_bytes,
            apply_mask=apply_mask,
            generate_heatmap=generate_heatmap,
            threshold=threshold
        )

    result["filename"] = filename
    result["is_sample"] = True
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
