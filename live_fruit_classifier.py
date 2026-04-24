import cv2
import numpy as np
import argparse
import time

# -------------------- ARGUMENT PARSING --------------------
parser = argparse.ArgumentParser(description="Live Fruit Freshness Classifier")
parser.add_argument("--model", type=str, default="fruit_freshness_classifier.h5",
                    help="Path to the trained Keras model (.h5)")
parser.add_argument("--fresh-img", type=str, default="fresh.jpeg",
                    help="Path to the fresh overlay image")
parser.add_argument("--rotten-img", type=str, default="rotten.jpeg",
                    help="Path to the rotten overlay image")
parser.add_argument("--camera", type=int, default=0,
                    help="Camera index (default: 0)")
parser.add_argument("--confidence-label", type=float, default=70.0,
                    help="Minimum confidence (%) to show label (default: 70)")
parser.add_argument("--confidence-high", type=float, default=80.0,
                    help="Confidence (%) threshold for green box (default: 80)")
parser.add_argument("--confidence-overlay", type=float, default=87.0,
                    help="Confidence (%) threshold to trigger overlay (default: 87)")
args = parser.parse_args()

# -------------------- SETUP --------------------
# Lazy-import TensorFlow so argparse --help works without GPU init delay
from tensorflow.keras.models import load_model

model = load_model(args.model)
print("✅ Model loaded successfully!")

class_names = [
    "Apple Fresh", "Apple Rotten",
    "Banana Fresh", "Banana Rotten",
    "Bellpepper Fresh", "Bellpepper Rotten",
    "Carrot Fresh", "Carrot Rotten",
    "Cucumber Fresh", "Cucumber Rotten",
    "Grape Fresh", "Grape Rotten",
    "Guava Fresh", "Guava Rotten",
    "Jujube Fresh", "Jujube Rotten",
    "Mango Fresh", "Mango Rotten",
    "Orange Fresh", "Orange Rotten",
    "Pomegranate Fresh", "Pomegranate Rotten",
    "Potato Fresh", "Potato Rotten",
    "Strawberry Fresh", "Strawberry Rotten",
    "Tomato Fresh", "Tomato Rotten"
]

# Load overlay images (optional — app still works without them)
fresh_img_orig = cv2.imread(args.fresh_img)
rotten_img_orig = cv2.imread(args.rotten_img)
if fresh_img_orig is None or rotten_img_orig is None:
    print("⚠️  Warning: overlay images not found — overlays will be skipped.")

# -------------------- CAMERA --------------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"❌ Error: Cannot access camera index {args.camera}.")
    exit(1)

print("🎥 Press 'q' to quit")

show_images_until = 0
current_status = None  # "fresh" or "rotten"

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Failed to grab frame")
        break

    frame_height, frame_width = frame.shape[:2]

    # ---- Resize overlay to 25% of frame size (relative, not hardcoded) ----
    overlay_w = max(100, frame_width // 4)
    overlay_h = max(100, frame_height // 4)
    if fresh_img_orig is not None:
        fresh_img = cv2.resize(fresh_img_orig, (overlay_w, overlay_h))
    if rotten_img_orig is not None:
        rotten_img = cv2.resize(rotten_img_orig, (overlay_w, overlay_h))

    # ---- Object detection via contours ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours largest-first so we evaluate the most prominent object
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_confidence = 0
    best_box = None
    best_label = "Detecting..."

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            break  # All remaining contours are smaller — stop early

        x, y, w, h = cv2.boundingRect(contour)

        # Skip bounding boxes that are nearly the whole frame (likely noise)
        if w > frame_width * 0.9 or h > frame_height * 0.9:
            continue

        roi = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (224, 224))
        roi_normalized = roi_resized / 255.0
        roi_expanded = np.expand_dims(roi_normalized, axis=0)

        preds = model.predict(roi_expanded, verbose=0)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        if confidence > args.confidence_label:
            best_box = (x, y, w, h)
            best_confidence = confidence
            best_label = class_names[class_index]
            break  # Highest-confidence large contour found; stop

    # ---- Draw bounding box and label ----
    if best_box:
        x, y, w, h = best_box

        if best_confidence >= args.confidence_high:
            box_color = (0, 255, 0)   # Green — high confidence
            label_text = f"{best_label} | {best_confidence:.1f}%"
        else:
            box_color = (0, 255, 255) # Yellow — lower confidence
            label_text = f"Detecting... ({best_confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Label background for readability
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y - 10, lh + 5)
        cv2.rectangle(frame, (x, label_y - lh - 5), (x + lw + 5, label_y + 3),
                      box_color, -1)
        cv2.putText(frame, label_text, (x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # ---- Trigger overlay at confidence_overlay threshold ----
        if best_confidence >= args.confidence_overlay:
            if "Fresh" in best_label:
                current_status = "fresh"
            elif "Rotten" in best_label:
                current_status = "rotten"
            show_images_until = time.time() + 3

    # ---- Render overlay (top-left corner, within frame bounds) ----
    if time.time() < show_images_until and current_status is not None:
        if current_status == "fresh" and fresh_img_orig is not None:
            overlay = fresh_img
        elif current_status == "rotten" and rotten_img_orig is not None:
            overlay = rotten_img
        else:
            overlay = None

        if overlay is not None:
            fh, fw = overlay.shape[:2]
            fy, fx = 20, 20
            # Guard: only paste if overlay fits inside frame
            if fy + fh <= frame_height and fx + fw <= frame_width:
                frame[fy:fy + fh, fx:fx + fw] = overlay

    cv2.imshow("Fruit Freshness Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
