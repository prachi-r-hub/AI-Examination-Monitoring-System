import time
import os
import sys
import cv2
import base64
import numpy as np
from typing import Dict, List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to Python module path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import config
from detector import ExamDetector
from database import IncidentDatabase
from video_processor import ExamVideoProcessor
from evaluator import ResearchEvaluator
from backend.camera_manager import CameraManager
from backend.db import load_db, add_history, add_evidence, EVIDENCE_DIR

app = FastAPI(
    title="Examination Monitoring Control Room API",
    description="FastAPI Backend for Real-Time YOLOv8s Examination Monitoring & Restricted Object Detection",
    version="2.0.0"
)

# Enable CORS for React Frontend (Vite Dev Server & Production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Backend Services
camera_manager = CameraManager()
from tracker import IncidentTracker
live_tracker = IncidentTracker()
db = IncidentDatabase()
detector = ExamDetector()

# Mount Static File Directory for Captured Evidence Screenshots
evidence_dir = config.EVIDENCE_DIR
os.makedirs(evidence_dir, exist_ok=True)
app.mount("/evidence_files", StaticFiles(directory=evidence_dir), name="evidence_files")

# Mount Static File Directory for Built-in Sample Demonstration Videos
sample_videos_dir = os.path.join(proj_root, "sample_videos")
os.makedirs(sample_videos_dir, exist_ok=True)
app.mount("/sample_videos", StaticFiles(directory=sample_videos_dir), name="sample_videos")

# Mount Static Frontend Build Directory if present
frontend_dist = os.path.join(proj_root, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

# Pydantic Schemas for Request Validation
class IncidentUpdateRequest(BaseModel):
    review_status: Optional[str] = None
    notes: Optional[str] = None

class SettingsUpdateRequest(BaseModel):
    conf_threshold: Optional[float] = None
    required_consecutive_frames: Optional[int] = None
    missed_frame_tolerance: Optional[int] = None
    duration_threshold: Optional[float] = None
    head_rep_threshold: Optional[int] = None
    head_window_seconds: Optional[float] = None
    voice_alerts_enabled: Optional[bool] = None
    voice_cooldown_seconds: Optional[float] = None

# =====================================================================
# 1. LIVE MONITORING & WEBCAM ENDPOINTS (SINGLE CAMERA LIFECYCLE)
# =====================================================================

@app.get("/api/camera/stream")
def get_camera_stream():
    """
    HTTP MJPEG Live Video Stream. Returns continuous annotated frame stream from single CameraManager.
    """
    print("6. MJPEG stream requested")
    return StreamingResponse(
        camera_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.post("/api/camera/start")
def start_camera(camera_index: int = Query(0)):
    """
    Starts the SINGLE webcam VideoCapture handle.
    """
    print("1. START CAMERA endpoint received")
    success = camera_manager.start(camera_index=camera_index)
    if not success:
        print("Camera failed to open")
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Camera unavailable — Failed to access hardware webcam."}
        )
    return {"success": True, "message": "Camera started cleanly."}

@app.post("/api/camera/stop")
def stop_camera():
    """
    Stops the camera and cleanly releases the single VideoCapture handle.
    """
    print("[FASTAPI SERVER] POST /api/camera/stop received")
    camera_manager.stop()
    print("[FASTAPI SERVER] ✔ Camera stopped cleanly.")
    return {"success": True, "message": "Camera handle released cleanly."}

@app.get("/api/camera/status")
def get_camera_status():
    """
    Returns real-time live monitoring metrics, active detections table, and recent timeline.
    """
    return {
        "is_active": camera_manager.is_active,
        "is_running": camera_manager.is_running,
        "metrics": camera_manager.latest_metrics,
        "active_detections": camera_manager.latest_active_detections,
        "timeline": camera_manager.latest_timeline
    }

# =====================================================================
# 2. IMAGE DETECTION MODE ENDPOINTS
# =====================================================================

@app.post("/api/image/detect")
def detect_image(file: UploadFile = File(...), conf: float = Query(0.25), mode: str = Query("image")):
    import time
    """
    Runs YOLOv8s best.pt detection on a single uploaded image.
    Returns base64 encoded original & annotated images + detection payload.
    """
    contents = file.file.read()
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None or np.mean(image) < 1.0:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    raw_detections, annotated_frame = detector.detect_frame(image, conf=conf)

    # Encode BGR frames to JPEG base64 strings
    _, orig_buf = cv2.imencode(".jpg", image)
    _, ann_buf = cv2.imencode(".jpg", annotated_frame)

    orig_b64 = base64.b64encode(orig_buf).decode("utf-8")
    ann_b64 = base64.b64encode(ann_buf).decode("utf-8")

    formatted_detections = []
    for idx, d in enumerate(raw_detections, 1):
        formatted_detections.append({
            "id": idx,
            "class_name": d["object_name"],
            "category": d["category"],
            "confidence": round(d["confidence"], 3),
            "status_label": f"🟢 ALLOWED" if d["category"] == "ALLOWED" else "🔴 RESTRICTED",
            "bbox": d["bounding_box"],
            "center_point": [d.get("center_x", 0), d.get("center_y", 0)]
        })

    events = []
    if mode == "live":
        # Pass to incident tracker to get temporal confirmed events
        tracked_dets, frame_summary = live_tracker.process_frame_detections(
            raw_detections,
            timestamp=time.time(),
            frame_image=image,
            mode="live_incidents"
        )
        events = frame_summary.get("new_events", [])

    return {
        "filename": file.filename,
        "total_detections": len(formatted_detections),
        "allowed_count": sum(1 for d in formatted_detections if d["category"] == "ALLOWED"),
        "restricted_count": sum(1 for d in formatted_detections if d["category"] == "RESTRICTED"),
        "detections": formatted_detections,
        "events": events,
        "original_image_b64": f"data:image/jpeg;base64,{orig_b64}",
        "annotated_image_b64": f"data:image/jpeg;base64,{ann_b64}"
    }

# =====================================================================
# 3. VIDEO ANALYSIS MODE ENDPOINTS
# =====================================================================

@app.get("/api/sample_videos")
def get_sample_videos():
    """
    Returns list of built-in sample videos available for viva demonstration.
    """
    samples_dir = os.path.join(proj_root, "sample_videos")
    samples = [
        {
            "id": "sample_1",
            "title": "Sample 1: Restricted Object Detection",
            "filename": "sample_1_prohibited_object.mp4",
            "description": "Demonstrates detection of a restricted mobile phone appearing during an examination session.",
            "url": "/sample_videos/sample_1_prohibited_object.mp4"
        },
        {
            "id": "sample_2",
            "title": "Sample 2: Lateral Head Movement Pattern",
            "filename": "sample_2_head_movement.mp4",
            "description": "Demonstrates detection of repeated lateral candidate head orientation patterns.",
            "url": "/sample_videos/sample_2_head_movement.mp4"
        },
        {
            "id": "sample_3",
            "title": "Sample 3: Object Passing Behaviour",
            "filename": "sample_3_object_passing.mp4",
            "description": "Demonstrates spatial-temporal transfer detection of an item passing between candidate tracks.",
            "url": "/sample_videos/sample_3_object_passing.mp4"
        }
    ]
    
    valid_samples = []
    for s in samples:
        fpath = os.path.join(samples_dir, s["filename"])
        if os.path.exists(fpath):
            valid_samples.append(s)
            
    return {"total": len(valid_samples), "sample_videos": valid_samples}

@app.post("/api/video/analyze")
def analyze_video(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Query(None),
    frame_sample_rate: int = Query(1)
):
    """
    Processes uploaded pre-recorded video or built-in sample video frame-by-frame with ExamVideoProcessor.
    Returns full analysis metrics, deduplicated event timeline, extracted evidence screenshots, and session status.
    """
    processor = ExamVideoProcessor()
    target_video_path = None
    is_temp = False

    if sample_filename:
        # User selected a built-in sample video
        sample_path = os.path.join(proj_root, "sample_videos", os.path.basename(sample_filename))
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample video '{sample_filename}' not found.")
        target_video_path = sample_path
    elif file:
        # User uploaded a custom video
        temp_dir = os.path.join(proj_root, "scratch")
        os.makedirs(temp_dir, exist_ok=True)
        target_video_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{file.filename}")
        with open(target_video_path, "wb") as f:
            f.write(file.file.read())
        is_temp = True
    else:
        raise HTTPException(status_code=400, detail="Please upload a video file or select a built-in sample video.")

    try:
        report = processor.process_video(
            video_path=target_video_path,
            mode="video_incidents",
            frame_sample_rate=max(1, frame_sample_rate)
        )
    except Exception as e:
        if is_temp and os.path.exists(target_video_path):
            os.remove(target_video_path)
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {str(e)}")

    # Inject Evidence and History
    try:
        r_events = report.get("event_summary", {}).get("restricted_object_events", 0)
        add_history("Video Analysis", target_video_path, r_events, report.get("overall_status", "NORMAL"))
        
        for ev in report.get("evidence_screenshots", []):
            fname = "video_incidents/" + ev["evidence_url"].split("/")[-1]
            add_evidence(
                source_type="Video Analysis", 
                object_name=ev.get("object_name", "Object"), 
                category="RESTRICTED", 
                confidence=float(ev.get("confidence", 0))/100.0, 
                image_filename=fname
            )
    except Exception as e:
        print("Failed to save video history:", e)

    # Clean up temp upload file
    if is_temp and os.path.exists(target_video_path):
        try:
            os.remove(target_video_path)
        except Exception:
            pass

    return report

# =====================================================================
# 4. INCIDENTS & EVIDENCE DATABASE ENDPOINTS
# =====================================================================

@app.get("/api/incidents")
def get_incidents(status: Optional[str] = Query(None)):
    """
    Fetches all persistent incident records stored in SQLite database.
    """
    filter_val = None if status in [None, "All Statuses", "ALL"] else status
    incidents = db.get_all_incidents(review_status_filter=filter_val)
    return {"total": len(incidents), "incidents": incidents}

@app.patch("/api/incidents/{incident_id}")
def update_incident(incident_id: str, body: IncidentUpdateRequest):
    """
    Updates proctor review status ('Reviewed', 'Dismissed') and notes in SQLite DB.
    """
    if body.review_status:
        db.update_review_status(incident_id, body.review_status)
    if body.notes is not None:
        db.add_reviewer_note(incident_id, body.notes)

    updated_inc = db.get_incident_by_id(incident_id)
    if not updated_inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found.")

    return {"success": True, "incident": updated_inc}


@app.post("/api/evidence/upload")
def upload_evidence(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        os.makedirs(os.path.join(EVIDENCE_DIR, "manual"), exist_ok=True)
        fname = f"manual_{int(time.time())}.jpg"
        fpath = os.path.join(EVIDENCE_DIR, "manual", fname)
        with open(fpath, "wb") as f_out:
            f_out.write(content)
            
        # Add to evidence DB
        add_evidence("Manual Upload", "User Uploaded Image", "RESTRICTED", 1.0, f"manual/{fname}")
        add_history("Manual Upload", fname, 1, "RESTRICTED")
        return {"success": True, "filename": fname}
    except Exception as e:
        print(f"Error uploading evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evidence")
def get_evidence_gallery():
    """
    Returns evidence screenshot file entries stored in physical directory.
    """
    incidents = db.get_all_incidents()
    evidence_items = []
    for inc in incidents:
        rel_path = inc.get("evidence_path", "")
        file_exists = os.path.exists(rel_path) if rel_path else False
        file_name = os.path.basename(rel_path) if rel_path else ""
        
        evidence_items.append({
            "incident_id": inc["incident_id"],
            "timestamp": inc["time_formatted"],
            "event_type": inc["event_type"],
            "object_name": inc["object_name"],
            "risk_score": inc["risk_score"],
            "risk_level": inc["risk_level"],
            "file_exists": file_exists,
            "evidence_url": f"/evidence_files/{file_name}" if file_exists else None,
            "file_path": rel_path
        })
    return {"total": len(evidence_items), "evidence": evidence_items}

# =====================================================================
# 5. CONFIGURATION & RESEARCH EVALUATION ENDPOINTS
# =====================================================================

@app.get("/api/settings")
def get_settings():
    """
    Returns current centralized configuration parameters.
    """
    return {
        "conf_threshold": getattr(config, "DEFAULT_CONF_THRESHOLD", 0.25),
        "required_consecutive_frames": getattr(config, "DEFAULT_REQUIRED_CONSECUTIVE_FRAMES", 3),
        "missed_frame_tolerance": getattr(config, "DEFAULT_MISSED_FRAME_TOLERANCE", 1),
        "duration_threshold": getattr(config, "SHORT_DURATION_THRESHOLD", 2.0),
        "head_rep_threshold": getattr(config, "REQUIRED_HEAD_TURNS", 3),
        "head_window_seconds": getattr(config, "HEAD_MOVEMENT_WINDOW_SECONDS", 4.0),
        "voice_alerts_enabled": getattr(config, "ENABLE_VOICE_ALERTS", True),
        "voice_cooldown_seconds": getattr(config, "VOICE_ALERT_COOLDOWN_SECONDS", 10.0),
        "gpu_device": "NVIDIA GeForce GTX 1050 (4GB)"
    }

@app.post("/api/settings")
def update_settings(body: SettingsUpdateRequest):
    """
    Validates and updates system parameters.
    """
    if body.conf_threshold is not None:
        config.DEFAULT_CONF_THRESHOLD = body.conf_threshold
    if body.required_consecutive_frames is not None:
        config.DEFAULT_REQUIRED_CONSECUTIVE_FRAMES = body.required_consecutive_frames
    if body.missed_frame_tolerance is not None:
        config.DEFAULT_MISSED_FRAME_TOLERANCE = body.missed_frame_tolerance
    if body.duration_threshold is not None:
        config.SHORT_DURATION_THRESHOLD = body.duration_threshold
    if body.head_rep_threshold is not None:
        config.REQUIRED_HEAD_TURNS = body.head_rep_threshold
    if body.head_window_seconds is not None:
        config.HEAD_MOVEMENT_WINDOW_SECONDS = body.head_window_seconds
    if body.voice_alerts_enabled is not None:
        config.ENABLE_VOICE_ALERTS = body.voice_alerts_enabled
        camera_manager.voice_manager.toggle(body.voice_alerts_enabled)
    if body.voice_cooldown_seconds is not None:
        config.VOICE_ALERT_COOLDOWN_SECONDS = body.voice_cooldown_seconds
        camera_manager.voice_manager.cooldown_seconds = body.voice_cooldown_seconds

    return {"success": True, "message": "Centralized configuration updated successfully."}

@app.post("/api/evaluator/run")
def run_evaluation():
    """
    Runs model evaluation and ablation study experiments for research reporting.
    """
    evaluator = ResearchEvaluator(detector=detector, db=db)
    det_eval = evaluator.evaluate_object_detection()
    exp1 = evaluator.run_experiment_1_single_vs_consecutive()
    exp2 = evaluator.run_experiment_2_confidence_thresholds()
    exp3 = evaluator.run_experiment_3_persistence_thresholds()
    exp4 = evaluator.run_experiment_4_duration_thresholds()

    return {
        "object_detection_eval": det_eval,
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4
    }

@app.get("/api/export/csv")
def export_csv():
    """
    Exports all incident records from SQLite database to a downloadable CSV file.
    """
    evaluator = ResearchEvaluator(db=db)
    csv_filepath = evaluator.export_incidents_to_csv()
    return FileResponse(
        path=csv_filepath,
        filename="examination_incidents_export.csv",
        media_type="text/csv"
    )

# Fallback root HTML response if frontend dist is built
@app.get("/")
def read_root():
    index_html = os.path.join(proj_root, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {
        "status": "online",
        "app": "Examination Monitoring Control Room API",
        "docs_url": "http://localhost:8000/docs",
        "message": "FastAPI backend running cleanly. Launch frontend React app on http://localhost:5173 or run build."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.post("/api/image/analyze_full")
def analyze_full_image(file: UploadFile = File(...)):
    import time
    contents = file.file.read()
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    raw_detections, annotated_frame = detector.detect_frame(image, conf=0.25)
    
    _, orig_buf = cv2.imencode(".jpg", image)
    _, ann_buf = cv2.imencode(".jpg", annotated_frame)

    orig_b64 = base64.b64encode(orig_buf).decode("utf-8")
    ann_b64 = base64.b64encode(ann_buf).decode("utf-8")
    
    payload = []
    restricted = 0
    fname = f"img_{int(time.time())}.jpg"
    
    for d in raw_detections:
        bbox = d.get("bounding_box", [0, 0, 0, 0])
        status = d.get("status", "UNKNOWN")
        if status == "RESTRICTED":
            restricted += 1
            
        payload.append({
            "class_name": str(d.get("class_name", "Unknown")),
            "category": status,
            "confidence": d.get("confidence", 0.0),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        })
        
    if restricted > 0:
        path = os.path.join(EVIDENCE_DIR, fname)
        with open(path, "wb") as f:
            f.write(ann_buf)
            
        for d in payload:
            if d["category"] == "RESTRICTED":
                add_evidence("Image Analysis", d["class_name"], d["category"], d["confidence"], fname)
                
    add_history("Image Analysis", file.filename, restricted, "REVIEW RECOMMENDED" if restricted > 0 else "NORMAL")
    
    return JSONResponse({
        "success": True,
        "annotated_image_b64": "data:image/jpeg;base64," + ann_b64,
        "original_image_b64": "data:image/jpeg;base64," + orig_b64,
        "detections": payload
    })


@app.post("/api/evidence/upload")
def upload_evidence(file: UploadFile = File(...)):
    try:
        content = file.file.read()
        os.makedirs(os.path.join(EVIDENCE_DIR, "manual"), exist_ok=True)
        fname = f"manual_{int(time.time())}.jpg"
        fpath = os.path.join(EVIDENCE_DIR, "manual", fname)
        with open(fpath, "wb") as f_out:
            f_out.write(content)
            
        # Add to evidence DB
        add_evidence("Manual Upload", "User Uploaded Image", "RESTRICTED", 1.0, f"manual/{fname}")
        add_history("Manual Upload", fname, 1, "RESTRICTED")
        return {"success": True, "filename": fname}
    except Exception as e:
        print(f"Error uploading evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evidence")
def get_evidence():
    return load_db()

@app.get("/api/history")
def get_history():
    return load_db()


from pydantic import BaseModel
class LiveEvent(BaseModel):
    source_type: str
    object_name: str
    category: str
    confidence: float
    image_b64: str

@app.post("/api/evidence/add")
def push_evidence(event: LiveEvent):
    import time
    fname = f"live_{int(time.time())}.jpg"
    path = os.path.join(EVIDENCE_DIR, fname)
    
    import base64
    b64 = event.image_b64.split(",")[1] if "," in event.image_b64 else event.image_b64
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
        
    add_evidence(event.source_type, event.object_name, event.category, event.confidence, fname)
    return {"success": True}

class SessionEnd(BaseModel):
    source_type: str
    source_details: str
    restricted_count: int
    status: str

@app.post("/api/history/add")
def push_history(session: SessionEnd):
    add_history(session.source_type, session.source_details, session.restricted_count, session.status)
    return {"success": True}

@app.get("/health")
def health():
    return {"status": "ok"}
