import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
import config

def point_in_or_near_box(px: float, py: float, bbox: list, buffer_px: float = 80.0) -> bool:
    """Checks if point (px, py) lies inside or within buffer_px pixels of bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return (x1 - buffer_px) <= px <= (x2 + buffer_px) and (y1 - buffer_px) <= py <= (y2 + buffer_px)

def calculate_center_distance_points(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

class PersonTrack:
    """Represents a tracked person instance with associated restricted objects."""

    def __init__(self, person_id: int, bbox: list, frame_number: int, timestamp: float):
        self.person_id = person_id
        self.bounding_box = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.first_seen_frame = frame_number
        self.first_seen_time = timestamp
        self.last_seen_frame = frame_number
        self.last_seen_time = timestamp
        self.missed_frames = 0
        self.associated_restricted_objects = []

    def update(self, bbox: list, frame_number: int, timestamp: float):
        self.bounding_box = bbox
        self.center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self.last_seen_frame = frame_number
        self.last_seen_time = timestamp
        self.missed_frames = 0

    def to_dict(self):
        return {
            "person_id": self.person_id,
            "bounding_box": self.bounding_box,
            "center": (round(self.center[0], 1), round(self.center[1], 1)),
            "last_seen_frame": self.last_seen_frame,
            "last_seen_time": round(self.last_seen_time, 2) if isinstance(self.last_seen_time, (float, int)) else self.last_seen_time,
            "associated_restricted_objects": list(self.associated_restricted_objects)
        }

class PersonTracker:
    """
    Lightweight Modular Person Tracking & Spatial Association Module.
    Uses pretrained COCO model (yolov8s.pt) filtered specifically for class 0 (person).
    Can be easily toggled on/off for performance optimization.
    """

    def __init__(self, enabled: bool = None, model_path: str = None):
        self.enabled = enabled if enabled is not None else config.ENABLE_PERSON_TRACKING
        self.model_path = model_path or config.COCO_PERSON_MODEL_PATH
        self.model = None
        self.person_tracks = []
        self.next_person_id = 1
        self.max_missed_frames = 3

        if self.enabled and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Warning: Could not load person detector model from {self.model_path}: {e}")
                self.enabled = False

    def toggle(self, enable: bool):
        """Enable or disable person tracking on demand."""
        self.enabled = enable
        if self.enabled and self.model is None and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)

    def detect_and_track_people(self, frame: np.ndarray, conf: float = 0.35, frame_number: int = 1, timestamp: float = 0.0) -> list:
        """
        Detects people in frame and maintains spatial-temporal person_id tracks.
        """
        if not self.enabled or self.model is None or frame is None:
            return []

        # Run detection filtered specifically for COCO class 0 ('person') on CPU
        results = self.model.predict(source=frame, classes=[0], conf=conf, device='cpu', verbose=False)
        
        detected_person_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                b = [int(round(xyxy[0])), int(round(xyxy[1])), int(round(xyxy[2])), int(round(xyxy[3]))]
                detected_person_boxes.append(b)

        # Match detected person boxes to active person_tracks
        matched_track_ids = set()
        for bbox in detected_person_boxes:
            box_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            best_person = None
            best_dist = float('inf')

            for p_track in self.person_tracks:
                if p_track.person_id in matched_track_ids:
                    continue

                dist = calculate_center_distance_points(p_track.center, box_center)
                if dist < 200.0 and dist < best_dist:
                    best_dist = dist
                    best_person = p_track

            if best_person is not None:
                best_person.update(bbox, frame_number, timestamp)
                matched_track_ids.add(best_person.person_id)
            else:
                new_person = PersonTrack(self.next_person_id, bbox, frame_number, timestamp)
                self.next_person_id += 1
                self.person_tracks.append(new_person)
                matched_track_ids.add(new_person.person_id)

        # Update missed frames for unmatched person tracks
        surviving = []
        for p_track in self.person_tracks:
            if p_track.person_id not in matched_track_ids:
                p_track.missed_frames += 1
            if p_track.missed_frames <= self.max_missed_frames:
                surviving.append(p_track)

        self.person_tracks = surviving
        return [p.to_dict() for p in self.person_tracks]

    def associate_restricted_objects(self, person_dicts: list, restricted_detections: list) -> list:
        """
        Associates detected restricted objects with nearby people based on spatial proximity.
        """
        if not self.enabled or not person_dicts or not restricted_detections:
            return restricted_detections

        # Clear previous frame's associations on person tracks
        for p in self.person_tracks:
            p.associated_restricted_objects = []

        for det in restricted_detections:
            if det.get("category") != "RESTRICTED":
                continue

            cx = det.get("center_x", (det["bounding_box"][0] + det["bounding_box"][2]) / 2.0)
            cy = det.get("center_y", (det["bounding_box"][1] + det["bounding_box"][3]) / 2.0)
            obj_name = det.get("object_name", "Restricted_Item")

            closest_person = None
            min_dist = float('inf')

            for p_dict in person_dicts:
                bbox = p_dict["bounding_box"]
                if point_in_or_near_box(cx, cy, bbox, buffer_px=config.PERSON_PROXIMITY_BUFFER_PX):
                    dist = calculate_center_distance_points((cx, cy), p_dict["center"])
                    if dist < min_dist:
                        min_dist = dist
                        closest_person = p_dict["person_id"]

            if closest_person is not None:
                det["associated_person_id"] = closest_person
                det["status_label"] += f" (Person #{closest_person})"
                
                # Record object in person track object list
                for p_track in self.person_tracks:
                    if p_track.person_id == closest_person:
                        if obj_name not in p_track.associated_restricted_objects:
                            p_track.associated_restricted_objects.append(obj_name)
            else:
                det["associated_person_id"] = None

        return restricted_detections

    def reset(self):
        """Resets person tracking state."""
        self.person_tracks = []
        self.next_person_id = 1
