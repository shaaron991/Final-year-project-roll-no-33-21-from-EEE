
import cv2
import torch
import time
import pygame
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ==========================================
# 1. REGISTER CHECKPOINT CLASSES SAFELY
# ==========================================

from torch.nn import (
    Conv2d, MaxPool2d, Sequential, BatchNorm2d,
    ModuleList, SiLU, Upsample, BCEWithLogitsLoss
)

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import (
    Conv, Concat, Bottleneck, Detect, DFL, SPPF, C2f
)
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

safe_classes = [
    Conv2d,
    MaxPool2d,
    Sequential,
    BatchNorm2d,
    ModuleList,
    SiLU,
    Upsample,
    BCEWithLogitsLoss,
    DetectionModel,
    IterableSimpleNamespace,
    BboxLoss,
    v8DetectionLoss,
    TaskAlignedAssigner,

    (Conv, "ultralytics.nn.modules.conv.Conv"),
    (Concat, "ultralytics.nn.modules.conv.Concat"),
    (Bottleneck, "ultralytics.nn.modules.block.Bottleneck"),
    (Detect, "ultralytics.nn.modules.head.Detect"),
    (DFL, "ultralytics.nn.modules.block.DFL"),
    (SPPF, "ultralytics.nn.modules.block.SPPF"),
    (C2f, "ultralytics.nn.modules.block.C2f"),
]

torch.serialization.add_safe_globals(safe_classes)


# ==========================================
# 2. FILE PATHS AND MODEL CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"
ALARM_PATH = BASE_DIR / "alarm.wav"

CONFIDENCE_THRESHOLD = 0.45

# Class IDs from your pretrained model
DROWSY_CLASSES = {
    0: "Eyes Closed",
    1: "Eyes Closed - Head Left",
    2: "Eyes Closed - Head Right",
    4: "Head Down",
    8: "Yawning"
}

ALERT_CLASSES = {
    3: "Focused",
    5: "Head Up",
    6: "Seeing Left",
    7: "Seeing Right"
}

# Time-based thresholds (seconds)
EYE_CLOSED_SECONDS = 1.2
HEAD_DOWN_SECONDS = 1.5
YAWN_SECONDS = 1.0


# ==========================================
# 3. LOAD PRETRAINED MODEL
# ==========================================

print("Loading pretrained YOLO model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")
print("Classes:", model.names)


# ==========================================
# 4. INITIALIZE ALARM
# ==========================================

pygame.mixer.init()

if not ALARM_PATH.exists():
    raise FileNotFoundError(f"Alarm file not found: {ALARM_PATH}")

alarm = pygame.mixer.Sound(str(ALARM_PATH))


# ==========================================
# 5. INITIALIZE WEBCAM
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot access webcam. Check camera permissions."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# 6. TEMPORAL DETECTION VARIABLES
# ==========================================

eye_closed_start = None
head_down_start = None
yawn_start = None

alarm_playing = False

# Limit processing rate for smoother performance
last_process_time = 0
PROCESS_INTERVAL = 0.05


# ==========================================
# 7. REAL-TIME WEBCAM LOOP
# ==========================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        current_time = time.time()

        eye_detected = False
        head_down_detected = False
        yawn_detected = False

        detections = []

        # Run model inference
        if current_time - last_process_time >= PROCESS_INTERVAL:

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            last_process_time = current_time

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    label = model.names[class_id]

                    detections.append(
                        (class_id, confidence, x1, y1, x2, y2, label)
                    )

                    if class_id in [0, 1, 2]:
                        eye_detected = True

                    elif class_id == 4:
                        head_down_detected = True

                    elif class_id == 8:
                        yawn_detected = True

        # ==================================
        # 8. DRAW DETECTION BOXES
        # ==================================

        for detection in detections:

            class_id, confidence, x1, y1, x2, y2, label = detection

            if class_id in DROWSY_CLASSES:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = f"{label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # ==================================
        # 9. TRACK HOW LONG EACH EVENT LASTS
        # ==================================

        if eye_detected:
            if eye_closed_start is None:
                eye_closed_start = current_time
        else:
            eye_closed_start = None

        if head_down_detected:
            if head_down_start is None:
                head_down_start = current_time
        else:
            head_down_start = None

        if yawn_detected:
            if yawn_start is None:
                yawn_start = current_time
        else:
            yawn_start = None

        eye_drowsy = (
            eye_closed_start is not None
            and current_time - eye_closed_start >= EYE_CLOSED_SECONDS
        )

        head_drowsy = (
            head_down_start is not None
            and current_time - head_down_start >= HEAD_DOWN_SECONDS
        )

        yawn_drowsy = (
            yawn_start is not None
            and current_time - yawn_start >= YAWN_SECONDS
        )

        drowsy = eye_drowsy or head_drowsy or yawn_drowsy

        # ==================================
        # 10. DISPLAY DRIVER STATUS
        # ==================================

        if drowsy:

            status = "DROWSY - WAKE UP!"
            status_color = (0, 0, 255)

            if not alarm_playing:
                alarm.play(loops=-1)
                alarm_playing = True

        else:

            status = "ALERT - DRIVE SAFELY"
            status_color = (0, 255, 0)

            if alarm_playing:
                alarm.stop()
                alarm_playing = False

        cv2.putText(
            frame,
            status,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            3
        )

        cv2.putText(
            frame,
            "Press Q to exit",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Driver Drowsiness Detection",
            frame
        )

        # Exit when Q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    if alarm_playing:
        alarm.stop()

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    print("Driver Drowsiness Detection stopped.")
