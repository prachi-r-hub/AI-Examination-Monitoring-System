import time
import math
import config
from utils import calculate_event_risk_score, save_evidence_snapshot

def calculate_iou(boxA: list, boxB: list) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

def calculate_center_distance(boxA: list, boxB: list) -> float:
    """Calculates Euclidean distance between centers of two bounding boxes [x1, y1, x2, y2]."""
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0
    return math.sqrt((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2)

class Track:
    """Represents a spatial-temporal track for a restricted object instance with duration tracking."""

    def __init__(self, track_id: int, object_name: str, category: str, confidence: float, bounding_box: list, frame_number: int, timestamp: float):
        self.track_id = track_id
        self.object_name = object_name
        self.category = category
        self.confidence = confidence
        self.bounding_box = bounding_box
        
        self.first_seen_frame = frame_number
        self.first_seen_time = timestamp
        self.last_seen_frame = frame_number
        self.last_seen_time = timestamp
        
        self.consecutive_frames = 1
        self.missed_frames = 0
        self.confirmed = False
        self.incident_event_created = False
        self.associated_person_id = None

    @property
    def duration_seconds(self) -> float:
        """Calculates total duration in seconds that the object has been tracked."""
        if self.last_seen_time is None or self.first_seen_time is None:
            return 0.0
        return round(max(0.0, self.last_seen_time - self.first_seen_time), 2)

    def get_significance(self) -> tuple:
        """
        Determines non-accusatory monitoring significance based on continuous duration.
        Returns: (significance_level, label_text)
        """
        dur = self.duration_seconds
        if dur < config.SHORT_DURATION_THRESHOLD:
            return "LOW", config.MSG_BRIEF_PRESENCE
        elif dur < config.LONG_DURATION_THRESHOLD:
            return "MEDIUM", f"{config.MSG_RESTRICTED_HANDLING} ({dur:.1f}s)"
        else:
            return "HIGH", f"{config.MSG_EXTENDED_HANDLING} ({dur:.1f}s)"

    def update(self, confidence: float, bounding_box: list, frame_number: int, timestamp: float, person_id: int = None):
        """Updates track with new detection in current frame."""
        self.confidence = max(self.confidence, confidence)
        self.bounding_box = bounding_box
        self.last_seen_frame = frame_number
        self.last_seen_time = timestamp
        self.consecutive_frames += 1
        self.missed_frames = 0
        if person_id is not None:
            self.associated_person_id = person_id

    def mark_missed(self):
        """Increments missed frame counter when object is absent in frame."""
        self.missed_frames += 1

    def to_dict(self):
        significance_level, significance_label = self.get_significance()
        return {
            "track_id": self.track_id,
            "object_name": self.object_name,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box,
            "person_id": self.associated_person_id,
            "first_seen_frame": self.first_seen_frame,
            "first_seen_time": round(self.first_seen_time, 2) if isinstance(self.first_seen_time, (float, int)) else self.first_seen_time,
            "last_seen_frame": self.last_seen_frame,
            "last_seen_time": round(self.last_seen_time, 2) if isinstance(self.last_seen_time, (float, int)) else self.last_seen_time,
            "duration_seconds": self.duration_seconds,
            "monitoring_significance": significance_level,
            "significance_label": significance_label,
            "consecutive_frames": self.consecutive_frames,
            "missed_frames": self.missed_frames,
            "confirmed": self.confirmed
        }


class IncidentTracker:
    """
    Advanced Incident Tracker featuring:
    - Restricted-Object Handling Event Creation
    - Person ID association
    - Configurable consecutive-frame confirmation
    - Real-time duration tracking (first_seen_time, last_seen_time, duration_seconds)
    - Automated risk score calculation
    - Non-accusatory monitoring terminology
    - Support for both Live Monitoring and Video Analysis modes
    """

    def __init__(self, required_consecutive_frames: int = None, max_missed_frames: int = None, confirmation_frames: int = None):
        if required_consecutive_frames is None and confirmation_frames is not None:
            required_consecutive_frames = confirmation_frames

        self.required_consecutive_frames = (
            required_consecutive_frames if required_consecutive_frames is not None 
            else config.DEFAULT_REQUIRED_CONSECUTIVE_FRAMES
        )
        self.max_missed_frames = (
            max_missed_frames if max_missed_frames is not None 
            else config.DEFAULT_MISSED_FRAME_TOLERANCE
        )
        
        self.active_tracks = []
        self.confirmed_incidents = []
        self.total_frames_processed = 0
        self.next_track_id = 1

    def process_frame_detections(self, detections: list, frame_number: int = None, timestamp: float = None, frame_image=None, mode: str = "live_incidents"):
        """
        Processes frame detections through tracking, duration & restricted-object handling pipeline.
        
        Returns:
            processed_detections (list): Updated detections payload
            frame_summary (dict): Current frame status summary with event logs
        """
        self.total_frames_processed += 1
        current_frame = frame_number if frame_number is not None else self.total_frames_processed
        current_time = timestamp if timestamp is not None else time.time()

        processed_detections = []
        new_confirmed_incidents = []
        matched_track_ids = set()

        restricted_dets = [d for d in detections if d.get("category") == "RESTRICTED"]

        # Match restricted detections to active tracks
        for det in restricted_dets:
            object_name = det.get("object_name") or det.get("class_name", "Unknown")
            bbox = det.get("bounding_box", [0, 0, 0, 0])
            confidence = det.get("confidence", 0.0)
            person_id = det.get("associated_person_id")

            best_track = None
            best_score = -1.0

            for track in self.active_tracks:
                if track.track_id in matched_track_ids:
                    continue
                if track.object_name != object_name:
                    continue

                iou = calculate_iou(track.bounding_box, bbox)
                dist = calculate_center_distance(track.bounding_box, bbox)

                if iou >= config.SPATIAL_MATCH_IOU_THRESHOLD or dist <= config.SPATIAL_MATCH_CENTER_DISTANCE:
                    score = iou + (1.0 / (dist + 1.0))
                    if score > best_score:
                        best_score = score
                        best_track = track

            if best_track is not None:
                best_track.update(confidence, bbox, current_frame, current_time, person_id=person_id)
                matched_track_ids.add(best_track.track_id)
                active_track_ref = best_track
            else:
                new_track = Track(
                    track_id=self.next_track_id,
                    object_name=object_name,
                    category="RESTRICTED",
                    confidence=confidence,
                    bounding_box=bbox,
                    frame_number=current_frame,
                    timestamp=current_time
                )
                if person_id is not None:
                    new_track.associated_person_id = person_id
                self.next_track_id += 1
                self.active_tracks.append(new_track)
                matched_track_ids.add(new_track.track_id)
                active_track_ref = new_track

            # Check for incident confirmation threshold (Restricted-object handling condition)
            sig_level, sig_label = active_track_ref.get_significance()
            
            if active_track_ref.consecutive_frames >= self.required_consecutive_frames:
                active_track_ref.confirmed = True
                
                # Calculate transparent rule-based risk score & factors
                from risk_engine import RiskEngine
                risk_res = RiskEngine().calculate_incident_risk({
                    "event_type": "Restricted-object handling detected",
                    "category": "RESTRICTED",
                    "object": object_name,
                    "confidence": active_track_ref.confidence,
                    "duration": active_track_ref.duration_seconds
                })
                risk_score = risk_res["risk_score"]
                risk_level = risk_res["risk_level"]
                contributing_factors = risk_res["contributing_factors"]

                # Generate Restricted-object handling event EXACTLY ONCE per continuous track
                if not active_track_ref.incident_event_created:
                    active_track_ref.incident_event_created = True
                    
                    incident_record = {
                        "incident_id": len(self.confirmed_incidents) + 1,
                        "event_type": "Restricted-object handling detected",
                        "track_id": active_track_ref.track_id,
                        "person_id": active_track_ref.associated_person_id,
                        "object": object_name,
                        "object_name": object_name,
                        "category": "RESTRICTED",
                        "confidence": active_track_ref.confidence,
                        "bounding_box": bbox,
                        "first_seen": active_track_ref.first_seen_frame,
                        "last_seen": active_track_ref.last_seen_frame,
                        "first_seen_time": active_track_ref.first_seen_time,
                        "last_seen_time": active_track_ref.last_seen_time,
                        "duration": active_track_ref.duration_seconds,
                        "duration_seconds": active_track_ref.duration_seconds,
                        "timestamp": current_time,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "contributing_factors": contributing_factors,
                        "evidence_frame": "",
                        "monitoring_significance": sig_level,
                        "consecutive_frames": active_track_ref.consecutive_frames,
                        "confirmed": True,
                        "alert_message": f"Restricted-object handling detected ({object_name}, {active_track_ref.duration_seconds:.1f}s)"
                    }

                    # Save annotated evidence screenshot if frame provided
                    if frame_image is not None:
                        from evidence_manager import EvidenceManager
                        saved_path = EvidenceManager().capture_evidence(incident_record, current_frame=frame_image, mode=mode)
                        incident_record["evidence_frame"] = saved_path

                    # Persist incident to SQLite database
                    try:
                        from database import IncidentDatabase
                        db_id = IncidentDatabase().insert_incident(incident_record)
                        incident_record["db_incident_id"] = db_id
                        
                        # Also sync to JSON history for the frontend
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from backend.db import add_evidence, add_history
                        
                        if saved_path:
                            rel_path = os.path.relpath(saved_path, config.EVIDENCE_DIR).replace("\\", "/")
                            add_evidence(
                                source_type="Live" if mode == "live_incidents" else "Video",
                                object_name=incident_record.get("object_name", "Unknown"),
                                category=incident_record.get("category", "RESTRICTED"),
                                confidence=incident_record.get("confidence", 0.0),
                                image_filename=rel_path
                            )
                            
                        add_history(
                            source_type="Live" if mode == "live_incidents" else "Video",
                            source_details="Real-time Tracking" if mode == "live_incidents" else "Pre-recorded Analysis",
                            restricted_count=1,
                            status="REVIEW RECOMMENDED"
                        )
                        
                    except Exception as e:
                        print(f"Warning: Could not insert incident into database: {e}")

                    new_confirmed_incidents.append(incident_record)
                    self.confirmed_incidents.append(incident_record)

                else:
                    # Update active confirmed event record
                    for inc in self.confirmed_incidents:
                        if inc["track_id"] == active_track_ref.track_id:
                            inc["last_seen"] = active_track_ref.last_seen_frame
                            inc["last_seen_time"] = active_track_ref.last_seen_time
                            inc["duration"] = active_track_ref.duration_seconds
                            inc["duration_seconds"] = active_track_ref.duration_seconds
                            inc["risk_score"] = risk_score
                            inc["monitoring_significance"] = sig_level
                            inc["alert_message"] = f"Restricted-object handling detected ({object_name}, {active_track_ref.duration_seconds:.1f}s)"

            # Build enriched detection item
            det_item = {
                "object_name": object_name,
                "class_name": object_name,
                "category": "RESTRICTED",
                "confidence": confidence,
                "bounding_box": bbox,
                "person_id": active_track_ref.associated_person_id,
                "associated_person_id": active_track_ref.associated_person_id,
                "first_seen": active_track_ref.first_seen_frame,
                "last_seen": active_track_ref.last_seen_frame,
                "first_seen_time": active_track_ref.first_seen_time,
                "last_seen_time": active_track_ref.last_seen_time,
                "duration": active_track_ref.duration_seconds,
                "duration_seconds": active_track_ref.duration_seconds,
                "monitoring_significance": sig_level,
                "consecutive_frames": active_track_ref.consecutive_frames,
                "missed_frames": active_track_ref.missed_frames,
                "confirmed": active_track_ref.confirmed,
                "incident_confirmed": active_track_ref.confirmed,
                "timestamp": current_time,
                "frame_number": current_frame,
                "status_label": (
                    f"Restricted-object handling detected ({object_name}, P#{active_track_ref.associated_person_id or 'N/A'}, {active_track_ref.duration_seconds:.1f}s)" 
                    if active_track_ref.confirmed 
                    else f"RESTRICTED (Candidate: Frame {active_track_ref.consecutive_frames}/{self.required_consecutive_frames})"
                )
            }
            processed_detections.append(det_item)

        # Handle unmatched active tracks (missed frames tolerance)
        surviving_tracks = []
        for track in self.active_tracks:
            if track.track_id not in matched_track_ids:
                track.mark_missed()

            if track.missed_frames <= self.max_missed_frames:
                surviving_tracks.append(track)

        self.active_tracks = surviving_tracks

        # Process ALLOWED and UNKNOWN detections
        for det in detections:
            category = det.get("category", "UNKNOWN")
            if category == "ALLOWED":
                object_name = det.get("object_name") or det.get("class_name", "Unknown")
                processed_detections.append({
                    "object_name": object_name,
                    "class_name": object_name,
                    "category": "ALLOWED",
                    "confidence": det.get("confidence", 0.0),
                    "bounding_box": det.get("bounding_box", [0, 0, 0, 0]),
                    "person_id": None,
                    "associated_person_id": None,
                    "first_seen": current_frame,
                    "last_seen": current_frame,
                    "first_seen_time": current_time,
                    "last_seen_time": current_time,
                    "duration": 0.0,
                    "duration_seconds": 0.0,
                    "monitoring_significance": "NONE",
                    "consecutive_frames": 1,
                    "missed_frames": 0,
                    "confirmed": False,
                    "incident_confirmed": False,
                    "timestamp": current_time,
                    "frame_number": current_frame,
                    "status_label": "ALLOWED (Normal)"
                })

        # Build summary
        active_candidates = [t.to_dict() for t in self.active_tracks if not t.confirmed]
        active_confirmed_tracks = [t.to_dict() for t in self.active_tracks if t.confirmed]

        frame_summary = {
            "frame_number": current_frame,
            "timestamp": current_time,
            "total_detections": len(processed_detections),
            "allowed_count": sum(1 for d in processed_detections if d["category"] == "ALLOWED"),
            "restricted_count": len(restricted_dets),
            "active_tracks_count": len(self.active_tracks),
            "active_candidates": active_candidates,
            "active_confirmed_tracks": active_confirmed_tracks,
            "new_confirmed_incidents": new_confirmed_incidents,
            "total_confirmed_incidents_all_time": len(self.confirmed_incidents)
        }

        return processed_detections, frame_summary

    def reset(self):
        """Resets the incident tracker state for a new monitoring session."""
        self.active_tracks = []
        self.confirmed_incidents = []
        self.total_frames_processed = 0
        self.next_track_id = 1
