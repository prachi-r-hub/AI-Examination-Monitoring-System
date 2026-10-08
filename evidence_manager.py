import os
import cv2
import time
import numpy as np
import config

class FrameBuffer:
    """
    Maintains a rolling buffer of (frame_image, detections, confidence) for selecting the optimal evidence frame.
    """
    def __init__(self, max_size: int = 7):
        self.max_size = max_size
        self.buffer = []

    def add_frame(self, frame: np.ndarray, detections: list, timestamp: float, confidence: float):
        if frame is None:
            return
        self.buffer.append({
            "frame": frame.copy(),
            "detections": list(detections),
            "timestamp": timestamp,
            "confidence": confidence
        })
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def select_best_frame(self) -> dict:
        """Selects the frame with maximum detection confidence from buffer."""
        if not self.buffer:
            return None
        return max(self.buffer, key=lambda item: item.get("confidence", 0.0))

    def clear(self):
        self.buffer = []


class EvidenceManager:
    """
    Automatic Evidence Screenshot Capture Engine featuring:
    - Frame Buffer selection (chooses highest-confidence / optimal frame)
    - Full visual overlay annotation: bounding boxes, object names, confidences, timestamp, event type, risk level.
    - Automated directory management (evidence/live_incidents, evidence/video_incidents).
    - Sequential filename formatting (incident_001.jpg, incident_002.jpg).
    - Prevents frame spamming (saves exactly 1 evidence file per confirmed incident).
    - Safe error handling for unwritable/missing directories.
    """

    def __init__(self, evidence_dir: str = None):
        self.evidence_dir = evidence_dir or config.EVIDENCE_DIR
        self.live_dir = config.LIVE_INCIDENTS_DIR
        self.video_dir = config.VIDEO_INCIDENTS_DIR
        self._ensure_directories()
        self.incident_counter = 0

    def _ensure_directories(self):
        try:
            for p in [self.evidence_dir, self.live_dir, self.video_dir]:
                os.makedirs(p, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create evidence directory: {e}")

    def generate_annotated_evidence_frame(
        self,
        frame: np.ndarray,
        incident_record: dict,
        detector=None,
        people_dicts: list = None
    ) -> np.ndarray:
        """
        Draws visual annotations and overlay header bar onto frame.
        Contains: bounding boxes, object names, confidence, timestamp, event type, risk level.
        """
        if frame is None:
            return None

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Annotate bounding boxes using detector if available
        if detector is not None:
            dets = incident_record.get("detections", [incident_record])
            annotated = detector.annotate_frame(annotated, dets, people_dicts=people_dicts)
        else:
            # Draw primary incident bounding box if available
            bbox = incident_record.get("bounding_box")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cat = incident_record.get("category", "RESTRICTED")
                color = config.COLOR_RESTRICTED if cat == "RESTRICTED" else config.COLOR_ALLOWED
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{incident_record.get('object', 'Item')} {incident_record.get('confidence', 0.85):.2f}"
                cv2.putText(annotated, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 2. Draw Top Overlay Header Bar with Event Type, Risk Level, and Timestamp
        header_height = 40
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Header Text
        evt_type = incident_record.get("event_type", "Confirmed Incident")
        risk_level = incident_record.get("risk_level", "HIGH")
        risk_score = incident_record.get("risk_score", 65)
        t_stamp = incident_record.get("time_formatted", str(incident_record.get("timestamp", "")))

        # Draw Header Title Text
        header_text = f"EVENT: {evt_type} | RISK: {risk_score}/100 [{risk_level}] | TIME: {t_stamp}"
        cv2.putText(
            annotated,
            header_text,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return annotated

    def capture_evidence(
        self,
        incident_record: dict,
        frame_buffer: FrameBuffer = None,
        current_frame: np.ndarray = None,
        mode: str = "live_incidents",
        detector=None,
        people_dicts: list = None
    ) -> str:
        """
        Captures and saves a fully annotated evidence screenshot for a confirmed incident.
        
        Args:
            incident_record (dict): Incident dictionary
            frame_buffer (FrameBuffer): Optional frame buffer to select best frame
            current_frame (np.ndarray): Current frame if no buffer
            mode (str): 'live_incidents' or 'video_incidents'
            detector: ExamDetector instance for drawing annotations
            people_dicts (list): Person tracks
            
        Returns:
            file_path (str): Saved evidence screenshot path linked to incident
        """
        if not incident_record:
            return ""

        # Select frame to annotate: use best frame from buffer if available
        target_frame = None
        if frame_buffer is not None:
            best_item = frame_buffer.select_best_frame()
            if best_item is not None:
                target_frame = best_item["frame"]

        if target_frame is None:
            target_frame = current_frame

        if target_frame is None:
            return ""

        self.incident_counter += 1
        self._ensure_directories()

        target_dir = self.live_dir if mode == "live_incidents" else self.video_dir

        # Standard filename: incident_001.jpg, incident_002.jpg
        incident_num = incident_record.get("incident_id", self.incident_counter)
        if isinstance(incident_num, str) and not incident_num.isdigit():
            filename = f"incident_{self.incident_counter:03d}.jpg"
        else:
            filename = f"incident_{int(incident_num):03d}.jpg"

        file_path = os.path.join(target_dir, filename)

        try:
            # Generate annotated evidence image
            annotated = self.generate_annotated_evidence_frame(
                target_frame,
                incident_record,
                detector=detector,
                people_dicts=people_dicts
            )

            # Save JPEG screenshot
            cv2.imwrite(file_path, annotated)

            # Link evidence filepath directly to incident record
            incident_record["evidence_frame"] = file_path
            return file_path
        except Exception as e:
            print(f"Warning: Failed to save evidence screenshot to {file_path}: {e}")
            return ""
