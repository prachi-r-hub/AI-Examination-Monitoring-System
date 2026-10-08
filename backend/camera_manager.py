import os
import sys
import cv2
import time
import threading
import torch
import numpy as np
from typing import Dict, List, Optional, Generator

# Add project root to Python module path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import config
from detector import ExamDetector
from tracker import IncidentTracker
from person_tracker import PersonTracker
from voice_alert import VoiceAlertManager
from evidence_manager import EvidenceManager
from database import IncidentDatabase

class CameraManager:
    """
    Decoupled Real-Time Camera Manager enforcing EXACTLY ONE webcam VideoCapture handle.
    
    Architecture:
    - Thread 1 (_raw_capture_loop): Hardware frame grabber running at 30 FPS.
      Constantly updates latest_raw_frame and drops stale queued frames. Zero latency.
    - Thread 2 (_inference_loop): Controlled YOLOv8s inference engine (12-15 FPS).
      Executes YOLO, person association, spatial persistence, voice alerts, evidence,
      and DB logging on CUDA GPU. Encodes JPEG once.
    - Generator (generate_mjpeg_stream): Yields pre-encoded JPEG bytes to browser at 30 FPS.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, camera_index: int = 0):
        if self._initialized:
            return
        self._initialized = True

        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_active = False

        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()

        # Core Backend Modules (100% Reused)
        self.detector = ExamDetector()
        self.tracker = IncidentTracker()
        self.person_tracker = PersonTracker()
        self.voice_manager = VoiceAlertManager()
        self.evidence_manager = EvidenceManager()
        self.db = IncidentDatabase()

        # Frame Buffers
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_encoded_jpeg: Optional[bytes] = None
        self.raw_frame_id = 0
        self.processed_frame_id = 0

        # Lightweight Performance Metrics
        self.device_name = "CUDA (NVIDIA GeForce GTX 1050)" if torch.cuda.is_available() else "CPU"
        self.camera_fps = 0.0
        self.yolo_fps = 0.0
        self.stream_fps = 0.0
        self.dropped_frames_count = 0
        self.last_inference_ms = 0.0

        self.latest_metrics: Dict = {
            "fps": 0.0,
            "gpu_inference_ms": 0.0,
            "people_count": 0,
            "allowed_count": 0,
            "restricted_count": 0,
            "confirmed_incidents": 0,
            "monitoring_risk_score": 0,
            "monitoring_risk_level": "LOW",
            "status_label": "CAMERA OFFLINE",
            "status_color": "GRAY",
            "device": self.device_name
        }
        self.latest_active_detections: List[Dict] = []
        self.latest_timeline: List[Dict] = []

    def start(self, camera_index: Optional[int] = None) -> bool:
        """
        Starts the SINGLE webcam VideoCapture pipeline.
        """
        with self.frame_lock:
            if camera_index is not None:
                self.camera_index = camera_index

            print("2. CameraManager start called")

            if self.is_running and self.cap is not None and self.cap.isOpened():
                print("Camera is already running with active VideoCapture handle.")
                return True

            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    print("3. VideoCapture opened = False")
                    self.is_running = False
                    self.is_active = False
                    self.latest_metrics["status_label"] = "CAMERA OFFLINE (Hardware Access Failed)"
                    self.latest_metrics["status_color"] = "RED"
                    return False

            print(f"3. VideoCapture opened = {self.cap.isOpened()}")
            print(f"[PERFORMANCE LOG] Execution Device: {self.device_name}")

            self.is_running = True
            self.is_active = True
            self.raw_frame_id = 0
            self.processed_frame_id = 0
            self.dropped_frames_count = 0

            self.latest_metrics["status_label"] = "🟢 MONITORING STATUS: NORMAL — Real-Time Live Feed Active"
            self.latest_metrics["status_color"] = "GREEN"

            # Launch Dedicated Capture Thread (30 FPS)
            self.capture_thread = threading.Thread(target=self._raw_capture_loop, daemon=True)
            self.capture_thread.start()

            # Launch Controlled Inference Thread (15 FPS Target)
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()

            return True

    def stop(self):
        """
        Cleanly stops capture & inference threads and releases VideoCapture handle.
        """
        with self.frame_lock:
            self.is_running = False
            self.is_active = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_raw_frame = None
            self.latest_annotated_frame = None
            self.latest_encoded_jpeg = None
            self.latest_metrics["fps"] = 0.0
            self.latest_metrics["status_label"] = "CAMERA OFFLINE"
            self.latest_metrics["status_color"] = "GRAY"
            print("✔ VideoCapture handle cleanly released and processing threads stopped.")

    def _raw_capture_loop(self):
        """
        Thread 1: Hardware Camera Capture Loop.
        Grabs frames at maximum hardware webcam speed (30 FPS), constantly replacing
        latest_raw_frame to guarantee zero buffer lag.
        """
        t_prev = time.time()
        first_frame_logged = False

        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if not first_frame_logged:
                first_frame_logged = True
                print("4. First frame received = True")

            t_now = time.time()
            self.camera_fps = round(1.0 / max(t_now - t_prev, 1e-5), 1)
            t_prev = t_now

            # Guarantee 640x480 resolution safety for GTX 1050
            if frame.shape[1] > 640 or frame.shape[0] > 480:
                frame = cv2.resize(frame, (640, 480))

            with self.frame_lock:
                if self.latest_raw_frame is not None and self.raw_frame_id > self.processed_frame_id:
                    self.dropped_frames_count += 1
                self.latest_raw_frame = frame
                self.raw_frame_id += 1

            time.sleep(0.005)

    def _inference_loop(self):
        """
        Thread 2: Controlled YOLOv8s Inference & Logic Loop.
        Processes latest_raw_frame at ~15 FPS. Runs YOLO, person tracker, persistence tracking,
        voice alerts, evidence capture, DB logging, and single JPEG encoding.
        """
        t_prev = time.time()
        yolo_logged = False

        while self.is_running:
            frame_to_process = None
            current_frame_id = 0

            with self.frame_lock:
                if self.latest_raw_frame is not None and self.raw_frame_id > self.processed_frame_id:
                    frame_to_process = self.latest_raw_frame.copy()
                    current_frame_id = self.raw_frame_id

            if frame_to_process is None:
                time.sleep(0.01)
                continue

            t_start = time.time()

            # 1. Person Detection & Tracking
            people_dicts = self.person_tracker.detect_and_track_people(
                frame_to_process, conf=0.35, frame_number=current_frame_id, timestamp=round(t_start, 2)
            )

            # 2. YOLOv8s Custom Object Detection (best.pt)
            raw_detections, inference_time_ms = self.detector.detect_frame(
                frame_to_process, conf=config.DEFAULT_CONF_THRESHOLD, frame_number=current_frame_id, timestamp=round(t_start, 2)
            )
            self.last_inference_ms = round(inference_time_ms, 1)

            if not yolo_logged:
                yolo_logged = True
                print("5. YOLO processed frame")

            # 3. Spatial Association
            raw_detections = self.person_tracker.associate_restricted_objects(people_dicts, raw_detections)

            # 4. Spatial Persistence & Duration Tracking
            processed_dets, summary = self.tracker.process_frame_detections(
                raw_detections, frame_number=current_frame_id, timestamp=round(t_start, 2), frame_image=frame_to_process, mode="live_incidents"
            )

            # 5. Handle Newly Confirmed Incidents
            new_confirmed = summary.get("new_confirmed_incidents", [])
            for inc in new_confirmed:
                self.voice_manager.trigger_alert_for_incident(inc, current_time=t_start)
                self.evidence_manager.capture_evidence(inc, current_frame=frame_to_process, mode="live_incidents")
                self.db.insert_incident(inc)

            # 6. Annotate Frame with REAL Bounding Boxes
            annotated_frame = self.detector.annotate_frame(frame_to_process, processed_dets, people_dicts=people_dicts)

            # Single JPEG Encode
            ret, encoded_jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            t_end = time.time()
            self.yolo_fps = round(1.0 / max(t_end - t_prev, 1e-5), 1)
            t_prev = t_end

            # Calculate Risk Score
            max_risk_score = 0
            max_risk_level = "LOW"
            for det in processed_dets:
                if det["category"] == "RESTRICTED":
                    r_score = 30 + (15 if det["confidence"] >= 0.85 else 0)
                    if r_score > max_risk_score:
                        max_risk_score = r_score
                        max_risk_level = "MEDIUM" if r_score >= 30 else "LOW"

            if len(self.tracker.confirmed_incidents) > 0:
                last_inc = self.tracker.confirmed_incidents[-1]
                max_risk_score = max(max_risk_score, last_inc.get("risk_score", 0))
                max_risk_level = last_inc.get("risk_level", "LOW")

            # Determine Status Label
            if new_confirmed:
                status_label = f"⚠️ CONFIRMED INCIDENT ALERT: {new_confirmed[-1].get('alert_message', 'Restricted object confirmed')}"
                status_color = "RED"
            elif summary["restricted_count"] > 0:
                status_label = f"🟡 RESTRICTED OBJECT IN VIEW — Tracking persistence ({summary['restricted_count']} detected)"
                status_color = "AMBER"
            else:
                status_label = "🟢 MONITORING STATUS: NORMAL — Real-Time Live Feed Active"
                status_color = "GREEN"

            # Prepare Detections Payload
            active_dets_payload = []
            for idx, det in enumerate(processed_dets, 1):
                active_dets_payload.append({
                    "id": idx,
                    "object_name": det["object_name"],
                    "category": det["category"],
                    "confidence": round(det["confidence"] * 100, 1),
                    "status_label": det["status_label"],
                    "person_id": det.get("person_id", "N/A"),
                    "bounding_box": det["bounding_box"]
                })

            # Fetch Recent Timeline
            db_timeline = self.db.get_recent_incidents(limit=5)
            timeline_payload = []
            for r in db_timeline:
                timeline_payload.append({
                    "time": r["time_formatted"],
                    "event_type": r["event_type"],
                    "object_name": r["object_name"],
                    "person_id": r["person_id"],
                    "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"],
                    "review_status": r["review_status"]
                })

            with self.frame_lock:
                self.processed_frame_id = current_frame_id
                self.latest_annotated_frame = annotated_frame
                if ret:
                    self.latest_encoded_jpeg = encoded_jpeg.tobytes()
                self.latest_metrics = {
                    "fps": self.yolo_fps,
                    "camera_fps": self.camera_fps,
                    "gpu_inference_ms": self.last_inference_ms,
                    "dropped_frames": self.dropped_frames_count,
                    "people_count": len(people_dicts),
                    "allowed_count": summary["allowed_count"],
                    "restricted_count": summary["restricted_count"],
                    "confirmed_incidents": len(self.tracker.confirmed_incidents),
                    "monitoring_risk_score": max_risk_score,
                    "monitoring_risk_level": max_risk_level,
                    "status_label": status_label,
                    "status_color": status_color,
                    "device": self.device_name
                }
                self.latest_active_detections = active_dets_payload
                self.latest_timeline = timeline_payload

            # Control inference pace (~15 FPS target to keep GTX 1050 smooth)
            time.sleep(0.04)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yields pre-encoded JPEG bytes to browser HTTP StreamingResponse at 30 FPS.
        Does NOT trigger duplicate YOLO inference.
        """
        sent_logged = False
        while True:
            if not self.is_running or self.latest_encoded_jpeg is None:
                placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 30
                cv2.putText(placeholder, "CAMERA OFFLINE", (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2)
                cv2.putText(placeholder, "Click 'START CAMERA' to begin live stream", (140, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                _, encoded_jpeg = cv2.imencode(".jpg", placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_jpeg.tobytes() + b'\r\n')
                time.sleep(0.2)
                continue

            if not sent_logged:
                sent_logged = True
                print("6. MJPEG stream requested")
                print("7. Frames being sent to browser")

            with self.frame_lock:
                jpeg_bytes = self.latest_encoded_jpeg

            if jpeg_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS streaming yield
