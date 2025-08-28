import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from libcamera import Transform

# Basic settings
BUFFER_BASE = "./capture_buffer"
NUM_BUFFERS = 4
FRAME_WIDTH, FRAME_HEIGHT = 800, 480
CAPTURE_FPS = 30
VIDEO_DURATION = 0.5  # seconds
FRAMES_PER_VIDEO = int(CAPTURE_FPS * VIDEO_DURATION)

CURRENT_BUFFER_FILE = "current_buffer.txt"

# Create buffer directories
for i in range(NUM_BUFFERS):
    os.makedirs(os.path.join(BUFFER_BASE, f"buffer_{i}"), exist_ok=True)

# PiCamera2 settings
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    controls={"FrameRate": CAPTURE_FPS},
    transform=Transform(hflip=0, vflip=0)
)
picam2.configure(config)
picam2.start()

print("[INFO] Starting 4-buffer image capture... (Press Ctrl+C to stop)")

current_buffer = 0

try:
    while True:
        buffer_dir = os.path.join(BUFFER_BASE, f"buffer_{current_buffer}")
        
        # Delete previous image files
        for f in os.listdir(buffer_dir):
            os.remove(os.path.join(buffer_dir, f))

        # Record current buffer
        with open(CURRENT_BUFFER_FILE, "w") as f:
            f.write(str(current_buffer))

        start_time = time.time()

        # Capture and save frames at 30FPS
        for frame_idx in range(FRAMES_PER_VIDEO):
            frame = picam2.capture_array()
            filename = os.path.join(buffer_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(filename, frame)
            time.sleep(1.0 / CAPTURE_FPS)

        elapsed = time.time() - start_time
        print(f"[DEBUG] Actual save FPS: {FRAMES_PER_VIDEO / elapsed:.2f}")

        current_buffer = (current_buffer + 1) % NUM_BUFFERS

except KeyboardInterrupt:
    print("\n[INFO] Image capture stopped")

