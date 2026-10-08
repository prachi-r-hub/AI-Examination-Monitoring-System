import os
import glob
import cv2
import subprocess
import numpy as np
import imageio_ffmpeg as ffmpeg

def create_h264_mp4(image_path, output_mp4, duration_sec=4.0, fps=15.0, motion_type='pan'):
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read input image: {image_path}")
        return False
    
    h, w, _ = img.shape
    num_frames = int(duration_sec * fps)

    cmd = [
        ffmpeg_exe, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        output_mp4
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(num_frames):
        if motion_type == 'pan':
            dx = int(np.sin(i / 5.0) * 12)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(img, M, (w, h))
        elif motion_type == 'head':
            # Rapid left-right shift sequence to trigger HeadMovementDetector
            dx = int(np.sin(i / 1.5) * 35)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(img, M, (w, h))
        elif motion_type == 'pass':
            # Diagonal spatial displacement sequence to trigger ObjectPassingDetector
            dx = int(np.sin(i / 2.0) * 25)
            dy = int(np.cos(i / 2.0) * 18)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(img, M, (w, h))
        else:
            frame = img
        
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    sz = os.path.getsize(output_mp4)
    print(f"[CREATED] Created H.264 {output_mp4} ({sz} bytes)")
    return True

def create_all_sample_videos():
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_videos")
    os.makedirs(sample_dir, exist_ok=True)
    
    test_imgs = sorted(glob.glob("Classroom_Dataset/Merged_Dataset/test/images/*.jpg"))
    if not test_imgs:
        print("[ERROR] No dataset test images found to build sample videos.")
        return

    # Select distinct images for each sample video
    img_earbuds = next((img for img in test_imgs if "earbuds" in os.path.basename(img).lower()), test_imgs[0])
    img_notebook1 = next((img for img in test_imgs if "notebook" in os.path.basename(img).lower()), test_imgs[min(3, len(test_imgs)-1)])
    img_notebook2 = test_imgs[min(4, len(test_imgs)-1)]

    # 1. Sample 1 — Prohibited Object (Earbuds / Phone present)
    p1 = os.path.join(sample_dir, "sample_1_prohibited_object.mp4")
    create_h264_mp4(img_earbuds, p1, duration_sec=4.0, fps=15.0, motion_type='pan')

    # 2. Sample 2 — Repeated Head Movement (Head motion sequence)
    p2 = os.path.join(sample_dir, "sample_2_head_movement.mp4")
    create_h264_mp4(img_notebook1, p2, duration_sec=4.0, fps=15.0, motion_type='head')

    # 3. Sample 3 — Possible Object Passing (Spatial shift sequence)
    p3 = os.path.join(sample_dir, "sample_3_object_passing.mp4")
    create_h264_mp4(img_notebook2, p3, duration_sec=4.0, fps=15.0, motion_type='pass')

if __name__ == "__main__":
    create_all_sample_videos()
