import cv2
import sys
import time

def find_working_camera(indices=(0, 1, 2, 3)):
    """
    Probes camera indices in sequence (0, 1, 2, 3).
    Returns (cap, working_index) for the FIRST camera that successfully
    opens and returns a valid frame.
    Releases all invalid handles.
    """
    for idx in indices:
        print(f"[TESTING] USB camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Verify camera actually yields a valid frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[SUCCESS] Camera index {idx} opened and returned valid frame ({frame.shape[1]}x{frame.shape[0]}).")
                return cap, idx
            cap.release()
            print(f"[WARNING] Index {idx} opened but failed to capture a valid frame.")
        else:
            cap.release()
            print(f"[FAILED] Index {idx} failed to open.")

    print("\n[ERROR] No active USB camera device found on indexes 0, 1, 2, 3.")
    print("Please connect a physical USB webcam to your computer and try again.")
    return None, -1

def run_camera_stream():
    """
    Main Camera Loop:
    - Exactly ONE cv2.VideoCapture handle.
    - Continuous real video display via cv2.imshow.
    - Clean 'q' / ESC key stop mechanism.
    - Always releases camera and closes windows.
    - Shows clear error if no camera is available.
    """
    print("==================================================")
    print("USB CAMERA DETECTION & CONTINUOUS STREAM MODULE")
    print("==================================================")

    cap, working_idx = find_working_camera(indices=(0, 1, 2, 3))
    if cap is None:
        print("\n[ABORTED] Camera test aborted: No valid hardware camera available.")
        return False

    print(f"\n[STREAMING] Starting live stream from Camera Index {working_idx}.")
    print("Press 'q' or 'ESC' on the video window to stop.")
    print("==================================================")

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Camera frame read failed or device disconnected.")
                break

            frame_count += 1
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            # Overlay clean technical status bar
            status_line = f"USB Camera Index {working_idx} | FPS: {fps:.1f} | Frame: {frame_count}"
            cv2.putText(
                frame,
                status_line,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Display real video frame
            cv2.imshow(f"USB Camera Index {working_idx} - Press 'q' or ESC to Exit", frame)

            # Check key press (1ms delay keeps feed responsive without backlogs)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]: # 'q', 'Q', or ESC key
                print("\n[STOP] Stop key pressed by user.")
                break

            # Check if user closed the OpenCV window manually
            if cv2.getWindowProperty(f"USB Camera Index {working_idx} - Press 'q' or ESC to Exit", cv2.WND_PROP_VISIBLE) < 1:
                print("\n[STOP] Camera window closed by user.")
                break

    finally:
        print("[SHUTDOWN] Releasing cv2.VideoCapture handle...")
        if cap is not None:
            cap.release()
        print("[SHUTDOWN] Closing all OpenCV windows...")
        cv2.destroyAllWindows()
        print("[DONE] Camera shutdown complete.")

    return True

if __name__ == "__main__":
    run_camera_stream()
