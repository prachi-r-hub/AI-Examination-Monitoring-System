import os
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import config

class ExamDetector:
    """
    Core Object Detection Engine for Examination Monitoring.
    Wraps YOLOv8s custom model (best.pt) using CPU ONLY.
    Provides structured detection output and visual bounding box annotations.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = None, iou_threshold: float = None):
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold or config.DEFAULT_CONF_THRESHOLD
        self.iou_threshold = iou_threshold or config.DEFAULT_IOU_THRESHOLD

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained YOLO model weights file not found at: {self.model_path}. "
                "Ensure best.pt is present."
            )

        # Load YOLO model ONCE
        self.model = YOLO(self.model_path)
        
        # Build normalized lookup for class names
        raw_names = getattr(self.model, "names", config.CLASS_MAPPING)
        self.names = {}
        for k, v in raw_names.items():
            name_str = str(v).strip().lower()
            if name_str in ["6", "class_6", "unknown_class_6"]:
                name_str = "earbuds"
            self.names[int(k)] = name_str
            
        # Guarantee class index 6 explicitly maps to "earbuds"
        self.names[6] = "earbuds"

    def classify_status(self, class_id: int, class_name: str) -> str:
        """
        Classifies object into ALLOWED or RESTRICTED status.
        ALLOWED: notebook, paper, pen, pencil (IDs 0, 1, 2, 3)
        RESTRICTED: phone, smartwatch, earbuds (IDs 4, 5, 6)
        UNKNOWN: invalid or unknown class ID
        """
        name_lower = str(class_name).strip().lower()
        if name_lower in ["notebook", "paper", "pen", "pencil"]:
            return "ALLOWED"
        elif name_lower in ["phone", "earbuds", "smartwatch", "restricted", "prohibited"]:
            return "RESTRICTED"
        
        # Fallback check by class_id
        if class_id in [0, 1, 2, 3]:
            return "ALLOWED"
        elif class_id in [4, 5, 6]:
            return "RESTRICTED"
        
        return "UNKNOWN"

    # Alias for backward compatibility
    classify_class = classify_status

    def detect_frame(self, frame: np.ndarray, conf: float = None, iou: float = None, frame_number: int = None, timestamp: float = None):
        """
        Runs object detection on a single frame (numpy BGR image).
        
        Returns:
            detections (list of dict): Structured detection information
            annotated_frame (np.ndarray): Frame with green/red bounding box annotations
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid NumPy array image.")

        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold

        device_val = "0" if torch.cuda.is_available() else "cpu"

        # Run inference
        t0 = time.time()
        results = self.model.predict(
            source=frame,
            conf=conf_val,
            iou=iou_val,
            device=device_val,
            verbose=False
        )
        self.last_inference_ms = (time.time() - t0) * 1000.0
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                
                # Confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                # Class ID and Class Name (NEVER display numeric string '6')
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.names.get(class_id, "earbuds" if class_id == 6 else f"object_{class_id}")
                if str(class_name).strip() == "6":
                    class_name = "earbuds"
                
                x1_int, y1_int, x2_int, y2_int = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                
                # Calculate center coordinates
                center_x = round((x1_int + x2_int) / 2.0, 2)
                center_y = round((y1_int + y2_int) / 2.0, 2)
                
                # Determine status (ALLOWED / PROHIBITED)
                status = self.classify_status(class_id, class_name)
                
                detection_item = {
                    "bounding_box": [x1_int, y1_int, x2_int, y2_int],
                    "class_name": str(class_name),
                    "confidence": round(confidence, 4),
                    "status": status,
                    "category": status,  # Backward compatibility
                    "class_id": class_id,
                    "object_name": str(class_name),
                    "center_x": center_x,
                    "center_y": center_y,
                    "frame_number": frame_number,
                    "timestamp": timestamp
                }
                detections.append(detection_item)

        # Generate annotated frame
        annotated_frame = self.annotate_frame(frame, detections)
        return detections, annotated_frame

    def annotate_frame(self, frame: np.ndarray, detections: list, people_dicts: list = None) -> np.ndarray:
        """
        Draws color-coded bounding boxes and labels on frame.
        - Green (0, 255, 0) for ALLOWED
        - Red (0, 0, 255) for PROHIBITED
        """
        annotated = frame.copy()

        # Draw Person bounding boxes if available
        if people_dicts:
            for p in people_dicts:
                px1, py1, px2, py2 = p["bounding_box"]
                pid = p["person_id"]
                p_color = config.COLOR_PERSON
                cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, 2)
                p_label = f"Person #{pid}"
                cv2.putText(
                    annotated,
                    p_label,
                    (px1 + 5, max(py1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    p_color,
                    2,
                    cv2.LINE_AA
                )
        
        for det in detections:
            x1, y1, x2, y2 = det["bounding_box"]
            status = det.get("status", det.get("category", "PROHIBITED"))
            class_name = det["class_name"]
            conf = det["confidence"]
            person_id = det.get("associated_person_id")
            
            # Green for ALLOWED, Red for PROHIBITED
            if status == "ALLOWED":
                color = config.COLOR_ALLOWED  # Green (0, 255, 0)
            else:
                color = config.COLOR_PROHIBITED  # Red (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label text
            conf_pct = int(round(conf * 100))
            label = f"{class_name} {conf_pct}% | {status}"
            if person_id is not None:
                label += f" -> P#{person_id}"
            
            # Text size calculation for background box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position label box above bounding box
            text_y1 = max(y1 - text_height - 6, 0)
            text_y2 = text_y1 + text_height + 6
            text_x2 = x1 + text_width + 8

            # Draw label background box
            cv2.rectangle(annotated, (x1, text_y1), (text_x2, text_y2), color, -1)
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_y2 - baseline - 2),
                font,
                font_scale,
                config.COLOR_TEXT,
                thickness,
                lineType=cv2.LINE_AA
            )

        return annotated

    def predict_image(self, image_input, conf: float = None):
        """
        Convenience wrapper for image files (path string) or loaded image arrays.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not decode image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray image.")

        return self.detect_frame(frame, conf=conf)

