import os
import sys
import cv2
import time
import threading
import config
from detector import ExamDetector
from camera_driver import find_working_camera
from voice_alert import VoiceAlertManager

class LiveExamMonitor:
    """
    Standalone Real-Time Examination Monitoring Engine with Voice Alerts.
    Connects real USB camera to CPU-only YOLOv8s detector (best.pt).
    
    Features:
    - Thread 1 (Capture Loop): Hardware frame reader updating latest_raw_frame (0 queue backlog).
    - Thread 2 (Async YOLO Inference): Non-blocking CPU YOLO inference engine (target 5-8 FPS).
    - Thread 3 (Voice Alert Engine): Asynchronous pyttsx3 TTS worker with 4.0s cooldown for PROHIBITED objects.
    - Main Loop (Display & UI): Renders smooth 30 FPS camera feed with green/red bounding boxes & RED alert banner.
    """

    def __init__(self, conf_threshold: float = 0.25, voice_cooldown_seconds: float = 4.0):
        print("[INIT] Initializing YOLOv8s detector on CPU...")
        t0 = time.time()
        # Load best.pt model ONCE
        self.detector = ExamDetector(conf_threshold=conf_threshold)
        print(f"[INIT] YOLO detector loaded successfully in {time.time() - t0:.2f}s.")

        # Initialize local non-blocking Voice Alert Manager (4.0s cooldown)
        self.voice_manager = VoiceAlertManager(cooldown_seconds=voice_cooldown_seconds)

        self.running = False
        self.cap = None
        self.working_idx = -1

        # Frame buffer (always newest frame, 0 queue)
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()

        # Detections cache for non-blocking UI rendering
        self.latest_detections = []
        self.inference_lock = threading.Lock()
        self.is_inferring = False
        self.yolo_fps = 0.0

    def _capture_loop(self):
        """Thread 1: Hardware frame reader loop. Keeps only the latest raw frame."""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            with self.frame_lock:
                self.latest_raw_frame = frame  # Overwrite immediately: 0 backlog queue

    def _run_async_inference(self, frame_to_process):
        """Thread 2: Asynchronous CPU YOLO inference worker."""
        t0 = time.time()
        try:
            dets, _ = self.detector.detect_frame(frame_to_process)
            elapsed = time.time() - t0
            calc_fps = 1.0 / max(elapsed, 1e-5)

            with self.inference_lock:
                self.latest_detections = dets
                self.yolo_fps = calc_fps
        except Exception as err:
            print(f"[ERROR] YOLO inference error: {err}")
        finally:
            self.is_inferring = False

    def run(self):
        """Main display and monitoring loop."""
        print("==================================================")
        print("STARTING LIVE USB CAMERA + YOLO EXAM MONITOR WITH VOICE ALERTS")
        print("==================================================")

        # Probe and open first working camera (indices 0, 1, 2, 3)
        self.cap, self.working_idx = find_working_camera(indices=(0, 1, 2, 3))
        if self.cap is None:
            print("\n[ABORTED] Cannot start Live Exam Monitor: No working USB camera found.")
            return False

        self.running = True

        # Start hardware capture thread
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()

        print(f"\n[MONITOR] Live stream running on Camera Index {self.working_idx}.")
        print("[MONITOR] Bounding Box Colors: GREEN = ALLOWED | RED = PROHIBITED")
        print("[MONITOR] Voice Alert Message: 'Warning. Prohibited object detected.' (4.0s cooldown)")
        print("[MONITOR] Press 'q' or 'ESC' on video window to exit.")
        print("==================================================")

        prev_display_time = time.time()
        window_title = f"AI Examination Monitor (Camera #{self.working_idx}) - Press 'q' to Exit"

        try:
            while self.running:
                # 1. Fetch newest raw camera frame
                with self.frame_lock:
                    frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. Trigger asynchronous YOLO CPU inference if worker is idle
                if not self.is_inferring:
                    self.is_inferring = True
                    threading.Thread(
                        target=self._run_async_inference,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()

                # 3. Retrieve latest detection bounding boxes
                with self.inference_lock:
                    current_dets = list(self.latest_detections)
                    current_yolo_fps = self.yolo_fps

                # 4. Trigger non-blocking voice alert if any PROHIBITED object (phone, earbuds, smartwatch) is detected
                self.voice_manager.trigger_prohibited_alert(current_dets)

                # 5. Draw bounding boxes & labels on current camera frame
                annotated = self.detector.annotate_frame(frame, current_dets)

                # 6. Calculate display FPS and overlay status banner
                curr_display_time = time.time()
                display_fps = 1.0 / max(curr_display_time - prev_display_time, 1e-5)
                prev_display_time = curr_display_time

                # Count ALLOWED vs PROHIBITED objects currently detected
                allowed_cnt = sum(1 for d in current_dets if d.get("status") == "ALLOWED")
                prohibited_cnt = sum(1 for d in current_dets if d.get("status") == "PROHIBITED")

                if prohibited_cnt > 0:
                    banner_color = (0, 0, 220)  # Bright Red
                    status_str = f"WARNING: PROHIBITED OBJECT DETECTED ({prohibited_cnt})"
                else:
                    banner_color = (0, 160, 0)  # Green
                    status_str = "MONITORING STATUS: NORMAL"

                h, w, _ = annotated.shape
                cv2.rectangle(annotated, (0, 0), (w, 35), banner_color, -1)
                
                info_text = f"{status_str} | Video: {display_fps:.1f} FPS | YOLO CPU: {current_yolo_fps:.1f} FPS | Allowed: {allowed_cnt}"
                cv2.putText(
                    annotated,
                    info_text,
                    (12, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # 7. Display real video frame
                cv2.imshow(window_title, annotated)

                # 8. Non-blocking key check (keeps video window completely responsive)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[STOP] Stop key pressed by user.")
                    break

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[STOP] Video window closed by user.")
                    break

        finally:
            self.running = False
            self.voice_manager.shutdown()
            print("[SHUTDOWN] Releasing cv2.VideoCapture handle...")
            if self.cap:
                self.cap.release()
            print("[SHUTDOWN] Closing OpenCV windows...")
            cv2.destroyAllWindows()
            print("[DONE] Live Examination Monitor shutdown complete.")

        return True

def run_live_monitor():
    monitor = LiveExamMonitor()
    return monitor.run()

if __name__ == "__main__":
    run_live_monitor()
