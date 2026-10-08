import cv2
import time
import numpy as np
import config
from utils import save_evidence_snapshot

class HeadMovementDetector:
    """
    Lightweight Head Orientation & Repeated Movement Detector.
    Estimates coarse head orientation (FORWARD, LEFT, RIGHT) per person_id
    and detects repeated head turn sequences within a configurable time window.
    
    Rules:
    - Ignores isolated single head movements.
    - Requires N distinct head turns (e.g. 3) within time window (e.g. 4.0s).
    - Prevents duplicate events for a single continuous movement sequence.
    - Non-accusatory: Generates 'Possible repeated head movement' event (not proof of cheating).
    """

    def __init__(self, required_turns: int = None, time_window_seconds: float = None):
        self.required_turns = required_turns or config.REQUIRED_HEAD_TURNS
        self.time_window_seconds = time_window_seconds or config.HEAD_MOVEMENT_WINDOW_SECONDS
        
        # State tracking per person_id
        # person_id -> { "history": [(timestamp, orientation)], "event_created": bool, "last_event_time": float }
        self.person_states = {}

    def estimate_orientation(self, person_bbox: list, frame: np.ndarray = None) -> str:
        """
        Estimates coarse head orientation (FORWARD, LEFT, RIGHT) using head ROI geometry & brightness/symmetry.
        """
        if person_bbox is None or len(person_bbox) < 4:
            return "FORWARD"

        px1, py1, px2, py2 = person_bbox
        p_width = max(1, px2 - px1)
        p_height = max(1, py2 - py1)

        if frame is not None and frame.size > 0:
            # Crop upper 35% of person ROI (head/face area)
            hy1 = max(0, py1)
            hy2 = max(hy1 + 1, min(frame.shape[0], py1 + int(p_height * 0.35)))
            hx1 = max(0, px1)
            hx2 = max(hx1 + 1, min(frame.shape[1], px2))

            head_crop = frame[hy1:hy2, hx1:hx2]
            if head_crop.size > 0:
                gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY) if len(head_crop.shape) == 3 else head_crop
                h_w = gray_head.shape[1]
                
                # Split left half vs right half intensity/edge mass
                mid = h_w // 2
                left_half = gray_head[:, :mid]
                right_half = gray_head[:, mid:]

                if left_half.size > 0 and right_half.size > 0:
                    left_mean = float(np.mean(left_half))
                    right_mean = float(np.mean(right_half))
                    
                    diff_ratio = (right_mean - left_mean) / (max(left_mean, right_mean) + 1e-5)

                    if diff_ratio > config.HEAD_OFFSET_THRESHOLD:
                        return "RIGHT"
                    elif diff_ratio < -config.HEAD_OFFSET_THRESHOLD:
                        return "LEFT"

        return "FORWARD"

    def process_person_frame(self, person_id: int, person_bbox: list, frame_number: int, timestamp: float, frame_image=None) -> dict:
        """
        Processes a person frame, updates orientation sequence, and checks for repeated head movement.
        
        Returns:
            event_dict (dict or None): Returns event payload if repeated movement threshold is met.
        """
        if person_id is None:
            return None

        orientation = self.estimate_orientation(person_bbox, frame=frame_image)

        if person_id not in self.person_states:
            self.person_states[person_id] = {
                "history": [],
                "event_created": False,
                "last_event_time": 0.0
            }

        state = self.person_states[person_id]
        history = state["history"]

        # Clean history entries outside time window
        min_time = timestamp - self.time_window_seconds
        history = [entry for entry in history if entry[0] >= min_time]

        # Record new orientation if changed from last entry
        if not history or history[-1][1] != orientation:
            history.append((timestamp, orientation))

        state["history"] = history

        # Count distinct directional turns across state history
        turns_count = 0
        for i in range(len(history) - 1):
            if history[i][1] != history[i+1][1]:
                turns_count += 1

        # Check if threshold met
        if turns_count >= self.required_turns and not state["event_created"]:
            state["event_created"] = True
            state["last_event_time"] = timestamp

            from evidence_manager import EvidenceManager
            from risk_engine import RiskEngine
            
            risk_res = RiskEngine().calculate_incident_risk({
                "event_type": config.MSG_REPEATED_HEAD_MOVEMENT,
                "confidence": 0.85,
                "duration": round(timestamp - history[0][0], 2) if history else 2.0
            })

            temp_record = {
                "event_type": config.MSG_REPEATED_HEAD_MOVEMENT,
                "person_id": person_id,
                "object": "Head Movement",
                "risk_score": risk_res["risk_score"],
                "risk_level": risk_res["risk_level"],
                "time_formatted": time.strftime('%H:%M:%S', time.gmtime(timestamp)),
                "timestamp": round(timestamp, 2)
            }

            ev_path = ""
            if frame_image is not None:
                ev_path = EvidenceManager().capture_evidence(temp_record, current_frame=frame_image, mode="video_incidents")

            event_record = {
                "event_type": config.MSG_REPEATED_HEAD_MOVEMENT,
                "person_id": person_id,
                "object": "Repeated Head Movement",
                "orientation_sequence": [h[1] for h in history],
                "turns_count": turns_count,
                "confidence": 0.85,
                "duration": round(timestamp - history[0][0], 2) if history else 2.0,
                "duration_seconds": round(timestamp - history[0][0], 2) if history else 2.0,
                "timestamp": round(timestamp, 2),
                "evidence_frame": ev_path,
                "risk_score": risk_res["risk_score"],
                "risk_level": risk_res["risk_level"],
                "contributing_factors": risk_res["contributing_factors"],
                "alert_message": f"{config.MSG_REPEATED_HEAD_MOVEMENT}: Person #{person_id} performed {turns_count} head turns within {self.time_window_seconds:.1f}s."
            }

            try:
                from database import IncidentDatabase
                db_id = IncidentDatabase().insert_incident(event_record)
                event_record["db_incident_id"] = db_id
            except Exception as e:
                print(f"Warning: Could not insert head movement incident into DB: {e}")

            return event_record

        # Reset event_created flag if state returns to FORWARD continuously
        if state["event_created"] and (timestamp - state["last_event_time"]) > (self.time_window_seconds * 1.5):
            state["event_created"] = False

        return None

    def reset(self):
        """Resets head movement tracking state."""
        self.person_states = {}
