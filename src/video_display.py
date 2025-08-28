import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time
import os
import glob

# Load YOLOv8n (TFLite)
yolo_model = YOLO("models/num8n_1000_V3_full_integer_quant_edgetpu.tflite")

# Load LPRNet model (SavedModel format)
lpr_model = tf.saved_model.load("models/korean_lprnet_edgetpu_best_model")
infer = lpr_model.signatures["serving_default"]

# Load Korean font
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font_small = ImageFont.truetype(font_path, 32)  
font_big = ImageFont.truetype(font_path, 128)     

# LPRNet character list
char_list = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "가", "나", "다", "라", "마", "거", "너", "더", "러", "머",
    "버", "서", "어", "저", "고", "노", "도", "로", "모", "보",
    "소", "오", "조", "구", "누", "두", "루", "무", "부", "수",
    "우", "주", "하", "-", " "
]

# Frame processing setting
skip_frames = 10

# Buffer directory setting
buffer_index = 0
buffer_dir = f"capture_buffer/buffer_{buffer_index}"
image_files = sorted(glob.glob(os.path.join(buffer_dir, "*.jpg")))
image_index = 0

# Variables
frame_count = 0
cached_frame = None
cached_detections = None
overlay_image = None
overlay_start_time = None
overlay_duration = 10 
last_ocr_text = ""    

# For FPS measurement
frame_counter = 0
start_time = time.time()
displayed_fps = 0

# OpenCV window settings
cv2.namedWindow("License Plate Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("License Plate Recognition", 800, 800)
cv2.moveWindow("License Plate Recognition", 100, 100)

while True:
    # Buffer switching
    if image_index >= len(image_files):
        buffer_index = (buffer_index + 1) % 4
        buffer_dir = f"capture_buffer/buffer_{buffer_index}"
        image_files = sorted(glob.glob(os.path.join(buffer_dir, "*.jpg")))
        image_index = 0
        if not image_files:
            print(f"[DEBUG] {buffer_dir} is empty, waiting...")
            time.sleep(0.01)
            continue

    # Read frame
    frame_path = image_files[image_index]
    frame = cv2.imread(frame_path)
    image_index += 1

    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        print(f"[ERROR] Invalid frame read: {frame_path}")
        continue

    frame_count += 1

     # YOLO inference
    if frame_count % skip_frames == 0:
        try:
            results = yolo_model(frame)
        except Exception as e:
            print(f"[ERROR] Error during YOLO inference: {e}")
            continue

        detections = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2))

        if detections:
            cached_frame = frame.copy()
            cached_detections = detections
        else:
            cached_detections = None

     # Clone frame for display
    display_frame = frame.copy()

    # Draw YOLO boxes
    if cached_detections:
        for x1, y1, x2, y2 in cached_detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    current_time = time.time()

    # Overlay and OCR result (display for 10 seconds)
    if overlay_image is not None and overlay_start_time is not None:
        if current_time - overlay_start_time < overlay_duration:
            # Show overlay
            overlay_small = cv2.resize(overlay_image, (320, 320))
            h, w, _ = display_frame.shape
            oh, ow, _ = overlay_small.shape
            display_frame[h - oh:h, w - ow:w] = overlay_small

            # Show OCR result at top-center with border
            display_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(display_pil)
            if last_ocr_text != "":
                try:
                    bbox = font_big.getbbox(last_ocr_text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(last_ocr_text, font=font_big)

                center_x = (display_pil.width - text_width) // 2
                center_y = 30

                # Draw border 
                outline_range = 3
                for dx in range(-outline_range, outline_range + 1):
                    for dy in range(-outline_range, outline_range + 1):
                        if dx != 0 or dy != 0:
                            draw.text((center_x + dx, center_y + dy), last_ocr_text, font=font_big, fill=(255, 255, 255))

                draw.text((center_x, center_y), last_ocr_text, font=font_big, fill=(0, 0, 0))

            display_frame = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)
        else:
            overlay_image = None
            overlay_start_time = None
            last_ocr_text = ""

    # FPS update
    frame_counter += 1
    if current_time - start_time >= 1.0:
        displayed_fps = frame_counter / (current_time - start_time)
        frame_counter = 0
        start_time = current_time

    # Show FPS at top-left
    display_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(display_pil)
    fps_text = f"FPS: {displayed_fps:.2f}"
    draw.text((20, 20), fps_text, font=font_small, fill=(255, 0, 255))
    display_frame = cv2.cvtColor(np.array(display_pil), cv2.COLOR_RGB2BGR)

    # Display
    cv2.imshow("License Plate Recognition", display_frame)
    key = cv2.waitKey(6) & 0xFF  

    # Space bar → OCR
    if key == 32:
        if cached_frame is not None and cached_detections:
            x1, y1, x2, y2 = cached_detections[0]
            plate_img = cached_frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                print("No cropped image.")
                continue

           # Preprocessing
            plate_img_resized = cv2.resize(plate_img, (94, 24))
            plate_img_rgb = cv2.cvtColor(plate_img_resized, cv2.COLOR_BGR2RGB)
            plate_img_rgb = np.expand_dims(plate_img_rgb, axis=0).astype("float32") / 255.0
            plate_tensor = tf.convert_to_tensor(plate_img_rgb)

            try:
                predictions = infer(input_1=plate_tensor)
            except Exception as e:
                print(f"[ERROR] Error during LPR inference: {e}")
                continue

            pred_indices = np.argmax(predictions["softmax"].numpy()[0], axis=1)
            predicted_text = ""
            prev_index = -1
            for idx in pred_indices:
                if idx != prev_index and idx < len(char_list):
                    if char_list[idx] not in ["-", " "]:
                        predicted_text += char_list[idx]
                prev_index = idx

            # Save OCR result
            last_ocr_text = predicted_text
            overlay_start_time = time.time()

            # Prepare overlay image
            overlay_image = cached_frame.copy()
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            image_pil = Image.fromarray(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
            draw_overlay = ImageDraw.Draw(image_pil)
            draw_overlay.text((x1, y1 - 20), predicted_text, font=font_small, fill=(0, 255, 0))
            overlay_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(f"capture_{int(time.time())}.jpg", cached_frame)
        else:
            print("No cached detection result yet.")

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
