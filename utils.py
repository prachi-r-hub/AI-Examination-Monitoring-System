import os
import cv2
import time
import config

from risk_engine import RiskEngine

_default_risk_engine = RiskEngine()

def calculate_event_risk_score(object_name: str, duration_seconds: float, has_person: bool = False, confidence: float = 0.85, category: str = "RESTRICTED") -> int:
    """
    Calculates transparent monitoring risk score (0-100) using RiskEngine rules.
    NOTE: This is NOT a probability of cheating.
    """
    res = _default_risk_engine.calculate_incident_risk({
        "object": object_name,
        "duration": duration_seconds,
        "confidence": confidence,
        "category": category
    })
    return res["risk_score"]

def save_evidence_snapshot(frame, incident_record: dict, mode: str = "live_incidents") -> str:
    """
    Saves an evidence frame screenshot to the evidence directory.
    
    Args:
        frame (np.ndarray): BGR image frame
        incident_record (dict): Incident metadata dictionary
        mode (str): 'live_incidents' or 'video_incidents'
        
    Returns:
        file_path (str): Relative or absolute path to saved snapshot
    """
    if frame is None:
        return ""

    target_dir = config.LIVE_INCIDENTS_DIR if mode == "live_incidents" else config.VIDEO_INCIDENTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    incident_id = incident_record.get("incident_id", int(time.time()))
    obj_name = incident_record.get("object", incident_record.get("object_name", "restricted_item")).lower()
    t_stamp = str(incident_record.get("timestamp", "0")).replace(".", "_").replace(":", "-")

    filename = f"incident_{incident_id}_{obj_name}_{t_stamp}.jpg"
    file_path = os.path.join(target_dir, filename)

    try:
        # Save snapshot using OpenCV
        cv2.imwrite(file_path, frame)
        return file_path
    except Exception as e:
        print(f"Warning: Failed to save evidence snapshot: {e}")
        return ""
