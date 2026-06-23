from ultralytics import YOLO
from datetime import datetime
import cv2
import os
import time
import csv

from configROI import (
    VIDEO_PATH,
    ROI_X, ROI_Y, ROI_W, ROI_H,
    MIN_CONFIDENCE_COUNT,
    MIN_CONFIDENCE_SAVE,
    COOLDOWN_DETECTION,
    COOLDOWN_SAVE,
    COOLDOWN_PRINT,
    OUTPUT_DIR,
    CSV_PATH,
    VIDEO_OUTPUT_DIR,
    SAVE_CLIPS,
    PRE_EVENT_SECONDS,
    MIN_CONFIDENCE_CLIP
)

model = YOLO("yolov8n.pt")


def validate_config():
    if not 0 <= MIN_CONFIDENCE_COUNT <= 1:
        raise ValueError("MIN_CONFIDENCE_COUNT must be between 0 and 1")

    if not 0 <= MIN_CONFIDENCE_SAVE <= 1:
        raise ValueError("MIN_CONFIDENCE_SAVE must be between 0 and 1")

    if MIN_CONFIDENCE_SAVE < MIN_CONFIDENCE_COUNT:
        raise ValueError("MIN_CONFIDENCE_SAVE should not be lower than MIN_CONFIDENCE_COUNT")

    if COOLDOWN_DETECTION < 0 or COOLDOWN_SAVE < 0 or COOLDOWN_PRINT < 0:
        raise ValueError("Cooldown values must be positive numbers in seconds")

    if ROI_X < 0 or ROI_Y < 0:
        raise ValueError("ROI_X and ROI_Y cannot be negative.")

    if ROI_W <= 0 or ROI_H <= 0:
        raise ValueError("ROI_W and ROI_H must be greater than 0.")
    
    if not 0 <= MIN_CONFIDENCE_CLIP <= 1:
        raise ValueError("MIN_CONFIDENCE_CLIP must be between 0 and 1")
    
    if MIN_CONFIDENCE_CLIP < MIN_CONFIDENCE_SAVE:
        raise ValueError("MIN_CONFIDENCE_CLIP should not be lower than MIN_CONFIDENCE_SAVE")


def save_event_clip(video_buffer, clip_path, fps, width, height):
    if not video_buffer:
        print("Video buffer is empty")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Could not create clip file")
        return False

    frames_written = 0

    for buffered_frame in video_buffer:
        if buffered_frame is None:
            continue

        buffered_frame = cv2.resize(buffered_frame, (width, height))
        writer.write(buffered_frame)
        frames_written += 1

    writer.release()

    print(f"Frames written to clip: {frames_written}")

    return frames_written > 0


validate_config()

CONTOUR_THRESHOLD = int((ROI_W * ROI_H) * 0.003)

video = cv2.VideoCapture(VIDEO_PATH)

if video.isOpened():
    print(f"✅ Using video file: {VIDEO_PATH}")
else:
    print("⚠️ Video file not found. Attempting to open system camera...")

    video.release()
    video = cv2.VideoCapture(0)

    if video.isOpened():
        print("✅ Using system camera")
    else:
        print("❌ No video file or camera available")
        exit()

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "type",
            "confidence",
            "persons_count",
            "image_path"
        ])


FRAME_INTERVAL = 30
frame_buffer = []
video_buffer = []

fps = video.get(cv2.CAP_PROP_FPS)

if fps == 0 or fps is None:
    fps = 30

MAX_VIDEO_BUFFER = int(fps * PRE_EVENT_SECONDS)

last_detection = 0
last_save = 0
last_print = 0

ret, initial_frame = video.read()

if not ret or initial_frame is None:
    print("Could not read the first frame")
    video.release()
    exit()

frame_height, frame_width = initial_frame.shape[:2]
VIDEO_WIDTH = frame_width
VIDEO_HEIGHT = frame_height

exceeds_width = ROI_X + ROI_W > frame_width
exceeds_height = ROI_Y + ROI_H > frame_height

if exceeds_width and exceeds_height:
    raise ValueError(f"The ROI exceeds both the width ({frame_width}px) and height ({frame_height}px) "
        "of the video. Please adjust ROI_X, ROI_Y, ROI_W or ROI_H.")
elif exceeds_width:
    raise ValueError(f"The ROI exceeds the video width ({frame_width}px). "
        "Please adjust ROI_X or ROI_W.")
elif exceeds_height:
    raise ValueError(f"The ROI exceeds the video height ({frame_height}px). "
        "Please adjust ROI_Y or ROI_H.")

# Converting to grayscale improves motion detection accuracy
initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
initial_gray = cv2.GaussianBlur(initial_gray, (5, 5), 0)
frame_buffer.append(initial_gray)

video_buffer.append(initial_frame.copy())

while True:
    ret, frame = video.read()

    if not ret:
        break

    clean_frame = frame.copy()

    video_buffer.append(frame.copy())

    if len(video_buffer) > MAX_VIDEO_BUFFER:
        video_buffer.pop(0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    frame_buffer.append(gray)

    if len(frame_buffer) > FRAME_INTERVAL:
        frame_buffer.pop(0)

    cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (255, 0, 0), 2)

    cv2.putText(frame, "ROI", (ROI_X, ROI_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Current and previous ROI regions for motion comparison
    current_roi = gray[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]
    previous_roi = frame_buffer[0][ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Compute frame difference to detect motion
    difference = cv2.absdiff(previous_roi, current_roi)

    mean_diff = difference.mean()
    adaptive_threshold = max(20, min(60, mean_diff * 2))

    # Convert to pure black-and-white to better isolate motion areas
    _, threshold = cv2.threshold(difference, adaptive_threshold, 255, cv2.THRESH_BINARY)

    threshold = cv2.erode(threshold, None, iterations=1)   # Remove noise (white specks)
    threshold = cv2.dilate(threshold, None, iterations=3)  # Expand motion regions

    # Locate exact motion areas within the frame
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_motion_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < CONTOUR_THRESHOLD:
            continue

        large_motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)
        x += ROI_X
        y += ROI_Y

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Motion in ROI", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if large_motion_detected:
        roi_frame = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

        results = model(roi_frame, conf=MIN_CONFIDENCE_COUNT, classes=[0], verbose=False)

        now = time.time()
        detected_persons = list(results[0].boxes)

        if detected_persons and now - last_detection >= COOLDOWN_DETECTION:
            best_confidence = max(float(obj.conf) for obj in detected_persons)

            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "detection",
                    f"{best_confidence:.2f}",
                    len(detected_persons),
                    "-"
                ])

            last_detection = now

        if detected_persons and now - last_print >= COOLDOWN_PRINT:
            print(f"Persons in ROI: {len(detected_persons)} | "
                f"Time: {datetime.now().strftime('%H:%M:%S')}")
            last_print = now

        high_confidence_persons = [obj for obj in detected_persons
            if float(obj.conf) >= MIN_CONFIDENCE_SAVE]
        
        clip_confidence_persons = [obj for obj in detected_persons
            if float(obj.conf) >= MIN_CONFIDENCE_CLIP]

        if high_confidence_persons and now - last_save >= COOLDOWN_SAVE:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"person_{timestamp}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            cv2.imwrite(filepath, clean_frame)
            last_save = now

            if SAVE_CLIPS and clip_confidence_persons and len(video_buffer) >= MAX_VIDEO_BUFFER:
                clip_name = f"clip_{timestamp}.avi"
                clip_path = os.path.join(VIDEO_OUTPUT_DIR, clip_name)

                clip_saved = save_event_clip(
                    video_buffer.copy(),
                    clip_path,
                    fps,
                    VIDEO_WIDTH,
                    VIDEO_HEIGHT
                )

                if clip_saved:
                    print(f"🎬 Clip saved: {clip_name}")

            elif SAVE_CLIPS:
                if not clip_confidence_persons:
                    print("⚠️ Clip not saved: confidence below MIN_CONFIDENCE_CLIP")
            elif len(video_buffer) < MAX_VIDEO_BUFFER:
                print("⚠️ Clip not saved: insufficient video buffer")

            best_confidence = max(float(obj.conf) for obj in high_confidence_persons)

            print(f"✅ Image saved: {filename}")
            print(f"Detected at: {datetime.now().strftime('%H:%M:%S')} | "
                f"Confidence: {best_confidence:.2f} | "
                f"Persons: {len(high_confidence_persons)}")

            with open(CSV_PATH, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "saved_image",
                    f"{best_confidence:.2f}",
                    len(high_confidence_persons),
                    filepath
                    ])

        frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W] = results[0].plot()

    cv2.imshow("Surveillance System", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()