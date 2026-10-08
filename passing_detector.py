import math
import time
import numpy as np

import config
from utils import save_evidence_snapshot

def calculate_distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class ObjectPassingDetector:
    """
    Lightweight Spatial-Temporal Object / Paper Passing Detector.
    Detects spatial-temporal transfer patterns:
    Object near Person A -> Object moves toward Person B -> Object near Person B -> Object remains near Person B.
    
    Principles:
    - Conservative: Prefers no alert if confidence < PASSING_MIN_CONFIDENCE to prevent false accusations.
    - Non-accusatory: Generates 'Possible object passing — review recommended' event.
    - Modular & Disableable: Can be cleanly toggled on/off.
    """

    def __init__(
        self,
        enabled: bool = None,
        min_displacement_px: float = None,
        max_duration_seconds: float = None,
        min_confidence: float = None
    ):
        self.enabled = enabled if enabled is not None else config.ENABLE_OBJECT_PASSING_DETECTION
        self.min_displacement_px = min_displacement_px or config.PASSING_MIN_DISPLACEMENT_PX
        self.max_duration_seconds = max_duration_seconds or config.PASSING_MAX_DURATION_SECONDS
        self.min_confidence = min_confidence or config.PASSING_MIN_CONFIDENCE

        # Object trajectory state history:
        # object_name -> list of dicts: [{ timestamp, frame_number, center: (cx, cy), person_id, conf, bbox }]
        self.object_trajectories = {}
        self.triggered_passing_keys = set()

    def toggle(self, enable: bool):
        """Enables or disables object passing detection on demand."""
        self.enabled = enable

    def process_frame(
        self,
        detections: list,
        people_dicts: list = None,
        frame_number: int = 1,
        timestamp: float = 0.0,
        frame_image=None
    ) -> list:
        """
        Processes frame detections and checks for spatial-temporal object passing patterns.
        
        Returns:
            new_events (list): List of generated passing event records for current frame.
        """
        if not self.enabled or not detections:
            return []

        new_events = []

        # Target classes for object passing: Paper, Notebook, Phone
        target_classes = {"Paper", "Notebook", "Phone", "notebook", "paper", "phone"}

        for det in detections:
            obj_name = det.get("object_name") or det.get("class_name", "")
            if obj_name not in target_classes:
                continue

            conf = det.get("confidence", 0.0)
            
            # Low-confidence conservatism rule: Prefer no alert if confidence is below threshold
            if conf < self.min_confidence:
                continue

            bbox = det.get("bounding_box", [0, 0, 0, 0])
            cx = det.get("center_x", (bbox[0] + bbox[2]) / 2.0)
            cy = det.get("center_y", (bbox[1] + bbox[3]) / 2.0)
            person_id = det.get("associated_person_id")

            # Fallback person lookup if not pre-associated
            if person_id is None and people_dicts:
                closest_p = None
                min_p_dist = float('inf')
                for p in people_dicts:
                    px1, py1, px2, py2 = p["bounding_box"]
                    # Expand person bbox buffer
                    if (px1 - config.PERSON_PROXIMITY_BUFFER_PX) <= cx <= (px2 + config.PERSON_PROXIMITY_BUFFER_PX) and \
                       (py1 - config.PERSON_PROXIMITY_BUFFER_PX) <= cy <= (py2 + config.PERSON_PROXIMITY_BUFFER_PX):
                        dist = calculate_distance((cx, cy), p["center"])
                        if dist < min_p_dist:
                            min_p_dist = dist
                            closest_p = p["person_id"]
                person_id = closest_p

            # Maintain trajectory history for object
            if obj_name not in self.object_trajectories:
                self.object_trajectories[obj_name] = []

            traj = self.object_trajectories[obj_name]
            
            # Append current frame observation
            current_obs = {
                "timestamp": timestamp,
                "frame_number": frame_number,
                "center": (cx, cy),
                "person_id": person_id,
                "confidence": conf,
                "bbox": bbox
            }
            traj.append(current_obs)

            # Clean trajectory entries older than max_duration_seconds
            cutoff_time = timestamp - self.max_duration_seconds
            traj = [obs for obs in traj if obs["timestamp"] >= cutoff_time]
            self.object_trajectories[obj_name] = traj

            # Check spatial-temporal transfer pattern:
            # Look for an earlier observation (Person A) transferred to current observation (Person B)
            for prev_obs in traj[:-1]:
                person_a = prev_obs["person_id"]
                person_b = current_obs["person_id"]

                # Requirement 1: Person A and Person B must both be identified and distinct
                if person_a is not None and person_b is not None and person_a != person_b:
                    
                    # Requirement 2: Spatial displacement must meet min_displacement_px
                    dist = calculate_distance(prev_obs["center"], current_obs["center"])
                    if dist >= self.min_displacement_px:

                        # Requirement 3: Check that object remains near Person B for >= 2 frames (Step 4 verification)
                        recent_person_b_count = sum(1 for o in traj[-3:] if o["person_id"] == person_b)
                        if recent_person_b_count >= 2:

                            # Prevent duplicate events for the same transfer episode
                            transfer_key = f"passing_{obj_name}_P{person_a}_P{person_b}_{int(prev_obs['timestamp'])}"
                            if transfer_key not in self.triggered_passing_keys:
                                self.triggered_passing_keys.add(transfer_key)

                                start_t = prev_obs["timestamp"]
                                end_t = current_obs["timestamp"]
                                duration = round(max(0.1, end_t - start_t), 2)

                                from evidence_manager import EvidenceManager
                                from risk_engine import RiskEngine

                                risk_res = RiskEngine().calculate_incident_risk({
                                    "event_type": config.MSG_OBJECT_PASSING,
                                    "object": obj_name,
                                    "confidence": conf,
                                    "duration": duration,
                                    "person_id": f"P#{person_a} -> P#{person_b}"
                                })

                                temp_record = {
                                    "event_type": config.MSG_OBJECT_PASSING,
                                    "object": obj_name,
                                    "confidence": conf,
                                    "person_id": f"P#{person_a} -> P#{person_b}",
                                    "risk_score": risk_res["risk_score"],
                                    "risk_level": risk_res["risk_level"],
                                    "time_formatted": time.strftime('%H:%M:%S', time.gmtime(end_t)),
                                    "timestamp": round(end_t, 2)
                                }

                                ev_path = ""
                                if frame_image is not None:
                                    ev_path = EvidenceManager().capture_evidence(temp_record, current_frame=frame_image, mode="video_incidents")

                                event_record = {
                                    "event_type": config.MSG_OBJECT_PASSING,
                                    "object": obj_name,
                                    "person_a": person_a,
                                    "person_b": person_b,
                                    "person_id": f"P#{person_a} -> P#{person_b}",
                                    "start_time": round(start_t, 2),
                                    "end_time": round(end_t, 2),
                                    "duration": duration,
                                    "duration_seconds": duration,
                                    "confidence": round(conf, 4),
                                    "timestamp": round(end_t, 2),
                                    "evidence_frame": ev_path,
                                    "risk_score": risk_res["risk_score"],
                                    "risk_level": risk_res["risk_level"],
                                    "contributing_factors": risk_res["contributing_factors"],
                                    "alert_message": (
                                        f"{config.MSG_OBJECT_PASSING}: {obj_name} transferred from "
                                        f"Person #{person_a} to Person #{person_b} over {duration:.1f}s."
                                    )
                                }

                                try:
                                    from database import IncidentDatabase
                                    db_id = IncidentDatabase().insert_incident(event_record)
                                    event_record["db_incident_id"] = db_id
                                except Exception as e:
                                    print(f"Warning: Could not insert object passing incident into DB: {e}")

                                new_events.append(event_record)
                                break  # Create one event per frame per object

        return new_events

    def reset(self):
        """Resets object passing detector state."""
        self.object_trajectories = {}
        self.triggered_passing_keys = set()
