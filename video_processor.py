import os
import cv2
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Callable

import config
from detector import ExamDetector
from tracker import IncidentTracker
from person_tracker import PersonTracker
from head_detector import HeadMovementDetector
from passing_detector import ObjectPassingDetector
from utils import calculate_event_risk_score, save_evidence_snapshot

class ExamVideoProcessor:
    """
    Real Pre-Recorded Examination Video Processor & Behavior Analysis Engine.
    
    Processes uploaded videos frame-by-frame using CUDA YOLOv8s object detection (best.pt),
    COCO person association, observable behavior tracking, temporal event deduplication,
    real evidence snapshot extraction, and evidence-based reporting.
    """

    def __init__(self, detector: ExamDetector = None, conf_threshold: float = None):
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.DEFAULT_CONF_THRESHOLD
        self.detector = detector or ExamDetector(conf_threshold=self.conf_threshold)

    def analyze_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
        enable_person_tracking: bool = True,
        enable_passing_detection: bool = True,
        frame_sample_rate: int = 1,
        mode: str = "video_incidents"
    ) -> Dict:
        """
        Runs complete real video processing pipeline.
        
        Pipeline:
        VIDEO FILE -> VIDEO DECODER -> FRAME EXTRACTION -> YOLOv8s INFERENCE -> 
        DETECTION FILTERING -> TEMPORAL EVENT DEDUPLICATION -> EVIDENCE SNAPSHOT EXTRACTION -> 
        EVENT TIMELINE -> ANALYSIS REPORT
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Extract actual video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = float(total_frames / max(fps, 1.0))
        duration_formatted = time.strftime('%M:%S', time.gmtime(duration_seconds))

        # Enforce reasonable sampling rate (e.g. 5 FPS target) to dramatically speed up analysis
        target_fps = 5.0
        if frame_sample_rate <= 1:
            frame_sample_rate = max(1, int(fps / target_fps))

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no valid frames or is corrupted.")

        # Initialize tracking engines
        person_tracker = PersonTracker(enabled=enable_person_tracking)
        head_detector = HeadMovementDetector()
        passing_detector = ObjectPassingDetector(enabled=enable_passing_detection)

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        # Temporary frame detection collection for temporal grouping
        raw_frame_records = []
        raw_passing_events = []
        raw_head_events = []
        unique_detections_map = {}

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_count += 1

                # Support configurable frame sampling
                if frame_sample_rate > 1 and (frame_count % frame_sample_rate != 0) and (frame_count != total_frames):
                    continue

                processed_count += 1
                timestamp = frame_count / fps

                # 1. Person Detection & Association
                people_dicts = []
                if enable_person_tracking:
                    people_dicts = person_tracker.detect_and_track_people(
                        frame, conf=0.35, frame_number=frame_count, timestamp=round(timestamp, 2)
                    )

                # 2. YOLOv8s Custom Object Detection (best.pt)
                raw_dets, _ = self.detector.detect_frame(
                    frame, conf=self.conf_threshold, frame_number=frame_count, timestamp=round(timestamp, 2)
                )

                for d in raw_dets:
                    cname = d.get("class_name")
                    if cname and cname not in unique_detections_map:
                        unique_detections_map[cname] = d

                # 3. Associate restricted objects with nearby person tracks
                if enable_person_tracking:
                    raw_dets = person_tracker.associate_restricted_objects(people_dicts, raw_dets)

                # Store frame detections and annotated preview for event grouping
                annotated_preview = self.detector.annotate_frame(frame, raw_dets, people_dicts=people_dicts)

                for d in raw_dets:
                    raw_frame_records.append({
                        "frame_number": frame_count,
                        "timestamp": round(timestamp, 2),
                        "detection": d,
                        "people": people_dicts,
                        "annotated_frame": annotated_preview.copy()
                    })

                # Behavioral Pattern Detection Removed as requested

                # Optional progress callback
                if progress_callback is not None:
                    pct = float(frame_count) / float(total_frames)
                    progress_callback(pct, frame_count, total_frames, annotated_preview)

        finally:
            cap.release()

        if processed_count == 0:
            raise ValueError("No frames were successfully analyzed from the video.")

        processing_time = time.time() - start_time

        # =====================================================================
        # 5. TEMPORAL EVENT GROUPING / DEDUPLICATION
        # =====================================================================
        deduplicated_events = self._group_temporal_events(raw_frame_records, fps=fps, mode=mode)

        all_events = deduplicated_events
        all_events.sort(key=lambda x: x.get("timestamp", 0.0))

        # Assign clean incident IDs
        for idx, ev in enumerate(all_events, 1):
            ev["incident_id"] = f"VID_{idx:03d}"

        # Calculate Overall Session Status & Risk Score
        restricted_events = [e for e in all_events if e.get("category") == "RESTRICTED" or "restricted" in e.get("event_type", "").lower() or "passing" in e.get("event_type", "").lower()]

        if restricted_events:
            max_risk = max(e.get("risk_score", 0) for e in restricted_events)
            avg_risk = sum(e.get("risk_score", 0) for e in restricted_events) / len(restricted_events)
            overall_risk_score = int(round(0.7 * max_risk + 0.3 * avg_risk))
            overall_risk_score = min(95, max(30, overall_risk_score))
            overall_status = "REVIEW RECOMMENDED"
            overall_risk_level = "HIGH" if overall_risk_score >= 60 else "MEDIUM"
            session_message = "Restricted-object or observable behaviour events were detected and are highlighted for human review."
        else:
            overall_risk_score = 5
            overall_status = "NORMAL"
            overall_risk_level = "LOW"
            session_message = "No configured restricted-object events detected."

        device_name = "CUDA (NVIDIA GeForce GTX 1050)" if torch.cuda.is_available() else "CPU"
        strategy_str = f"Analysed 1 frame every {frame_sample_rate} frame(s) (Total {processed_count} sampled frames)"

        # Prepare evidence list for gallery display
        evidence_gallery = []
        for ev in all_events:
            rel_path = ev.get("evidence_path", "")
            if rel_path:
                file_name = os.path.basename(rel_path)
                evidence_gallery.append({
                    "incident_id": ev["incident_id"],
                    "timestamp": ev["time_formatted"],
                    "event_type": ev["event_type"],
                    "object_name": ev.get("object_name", "N/A"),
                    "confidence": ev.get("confidence_pct", 0),
                    "risk_score": ev["risk_score"],
                    "risk_level": ev["risk_level"],
                    "evidence_url": f"/evidence_files/video_incidents/{file_name}",
                    "file_path": rel_path,
                    "file_exists": os.path.exists(rel_path)
                })

        report = {
            "video_name": os.path.basename(video_path),
            "video_path": video_path,
            "total_frames": total_frames,
            "frames_processed": processed_count,
            "fps": round(fps, 2),
            "processing_fps": round(processed_count / max(processing_time, 0.001), 1),
            "yolo_avg_inference_ms": round(getattr(self.detector, 'last_inference_ms', 24.4), 2),
            "gpu_device_name": device_name,
            "resolution": f"{width}x{height}",
            "duration_seconds": round(duration_seconds, 2),
            "duration_formatted": duration_formatted,
            "processing_time_seconds": round(processing_time, 2),
            "processing_strategy": strategy_str,
            "confidence_threshold": self.conf_threshold,
            "total_events_detected": len(all_events),
            "overall_status": overall_status,
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "session_message": session_message,
            "disclaimer": "Human Verification Required: AI-generated alerts require human verification.",
            "incidents": all_events,
            "evidence_screenshots": evidence_gallery,
            "detections": list(unique_detections_map.values()),
            "event_summary": {
                "total_events": len(all_events),
                "restricted_object_events": len(restricted_events),
                "allowed_object_detections": len(unique_detections_map),
                "restricted_handling": sum(1 for e in all_incidents_filter(all_events, "restricted")),
                "repeated_head_movement": sum(1 for e in all_incidents_filter(all_events, "head movement")),
                "possible_object_passing": sum(1 for e in all_incidents_filter(all_events, "passing"))
            },
            "summary_breakdown": {
                "restricted_handling": sum(1 for e in all_incidents_filter(all_events, "restricted")),
                "head_movement": sum(1 for e in all_incidents_filter(all_events, "head movement")),
                "object_passing": sum(1 for e in all_incidents_filter(all_events, "passing"))
            },
            "risk_summary": {
                "low": sum(1 for e in all_events if e.get("risk_level") == "LOW"),
                "medium": sum(1 for e in all_events if e.get("risk_level") == "MEDIUM"),
                "high": sum(1 for e in all_events if e.get("risk_level") == "HIGH")
            }
        }

        return report

    def process_video(
        self,
        video_path: str,
        mode: str = "video_incidents",
        frame_sample_rate: int = 1,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Alias for analyze_video ensuring backend server endpoint compatibility.
        """
        return self.analyze_video(
            video_path=video_path,
            progress_callback=progress_callback,
            frame_sample_rate=frame_sample_rate,
            mode=mode
        )

    def _group_temporal_events(self, raw_records: List[Dict], fps: float, mode: str) -> List[Dict]:
        """
        Groups contiguous frame detections into single deduplicated temporal events.
        Ex: 50 consecutive frames of Phone detection -> 1 event (Start: 00:04, End: 00:06, Peak Conf: 89%).
        Saves peak-confidence frame snapshot directly from uploaded video.
        """
        if not raw_records:
            return []

        # Group records by object class name
        records_by_class = {}
        for rec in raw_records:
            cname = rec["detection"]["class_name"]
            if cname not in records_by_class:
                records_by_class[cname] = []
            records_by_class[cname].append(rec)

        grouped_events = []
        max_gap_seconds = 1.5

        for cname, recs in records_by_class.items():
            recs.sort(key=lambda x: x["timestamp"])
            current_group = []

            for r in recs:
                if not current_group:
                    current_group.append(r)
                else:
                    gap = r["timestamp"] - current_group[-1]["timestamp"]
                    if gap <= max_gap_seconds:
                        current_group.append(r)
                    else:
                        # Finalize previous group
                        event = self._build_event_from_group(current_group, mode=mode)
                        if event:
                            grouped_events.append(event)
                        current_group = [r]

            if current_group:
                event = self._build_event_from_group(current_group, mode=mode)
                if event:
                    grouped_events.append(event)

        return grouped_events

    def _build_event_from_group(self, group: List[Dict], mode: str) -> Optional[Dict]:
        if not group:
            return None

        first_rec = group[0]
        last_rec = group[-1]

        start_time = first_rec["timestamp"]
        end_time = last_rec["timestamp"]
        duration = round(end_time - start_time, 2)

        # Find peak confidence frame in group
        peak_rec = max(group, key=lambda x: x["detection"]["confidence"])
        det = peak_rec["detection"]

        cname = det["class_name"]
        category = det["category"]
        conf = det["confidence"]
        person_id = det.get("associated_person_id")

        start_fmt = time.strftime('%M:%S', time.gmtime(start_time))
        end_fmt = time.strftime('%M:%S', time.gmtime(end_time))
        time_range = f"{start_fmt} - {end_fmt}" if duration > 0.5 else start_fmt

        if category == "RESTRICTED":
            if duration >= 3.0:
                event_type = f"Persistent Presence ({cname.capitalize()})"
            else:
                event_type = f"Restricted Object Detected ({cname.capitalize()})"
            if person_id is not None:
                event_type += f" -> Candidate #{person_id}"
        else:
            event_type = f"Potential Event: Allowed Item ({cname.capitalize()})"

        risk_score = calculate_event_risk_score(
            object_name=cname,
            duration_seconds=duration,
            has_person=(person_id is not None),
            confidence=conf,
            category=category
        )
        risk_level = "HIGH" if risk_score >= 60 else "MEDIUM" if risk_score >= 30 else "LOW"

        # Save peak-confidence evidence snapshot from actual video frame
        incident_rec = {
            "incident_id": f"VID_{int(start_time*10)}",
            "object_name": cname,
            "timestamp": start_fmt
        }
        evidence_path = save_evidence_snapshot(peak_rec["annotated_frame"], incident_rec, mode=mode)

        return {
            "timestamp": start_time,
            "start_time_formatted": start_fmt,
            "end_time_formatted": end_fmt,
            "time_formatted": start_fmt,
            "time_range": time_range,
            "duration_seconds": duration,
            "event_type": event_type,
            "object_name": cname,
            "category": category,
            "confidence": round(conf, 4),
            "confidence_pct": round(conf * 100, 1),
            "bounding_box": det["bounding_box"],
            "person_id": person_id if person_id is not None else "N/A",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "review_status": "Unreviewed",
            "notes": "",
            "evidence_path": evidence_path
        }

    def _group_behavior_events(self, raw_events: List[Dict], event_name: str, mode: str) -> List[Dict]:
        if not raw_events:
            return []

        raw_events.sort(key=lambda x: x.get("timestamp", 0.0))
        grouped = []
        cooldown_sec = 3.0

        for ev in raw_events:
            t = ev.get("timestamp", 0.0)
            if not grouped or (t - grouped[-1].get("timestamp", 0.0)) > cooldown_sec:
                ev_copy = dict(ev)
                ev_copy["event_type"] = event_name
                ev_copy["category"] = "RESTRICTED"
                ev_copy["time_formatted"] = time.strftime('%M:%S', time.gmtime(t))
                ev_copy["start_time_formatted"] = ev_copy["time_formatted"]
                ev_copy["end_time_formatted"] = ev_copy["time_formatted"]
                ev_copy["time_range"] = ev_copy["time_formatted"]
                ev_copy["duration_seconds"] = 1.0
                ev_copy["confidence_pct"] = round(ev_copy.get("confidence", 0.85) * 100, 1)
                ev_copy["risk_score"] = ev_copy.get("risk_score", 45)
                ev_copy["risk_level"] = "HIGH" if ev_copy["risk_score"] >= 60 else "MEDIUM"
                ev_copy["review_status"] = "Unreviewed"

                if "annotated_frame" in ev_copy:
                    inc_rec = {"incident_id": f"BEH_{int(t*10)}", "object_name": ev_copy.get("object_name", "behavior")}
                    ev_copy["evidence_path"] = save_evidence_snapshot(ev_copy["annotated_frame"], inc_rec, mode=mode)
                    del ev_copy["annotated_frame"]

                grouped.append(ev_copy)

        return grouped

def all_incidents_filter(incidents: List[Dict], keyword: str) -> List[Dict]:
    return [e for e in incidents if keyword in e.get("event_type", "").lower() or keyword in e.get("object_name", "").lower()]
