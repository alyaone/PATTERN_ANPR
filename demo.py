import cv2
from ultralytics import YOLO
from collections import Counter, deque

from utils.ocr import read_plate

# =========================
# LOAD MODEL
# =========================

model = YOLO("best.pt")

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# OCR STABILIZER
# =========================q

ocr_buffer = deque(maxlen=15)

frame_counter = 0

print("ANPR STARTED")
print("Press Q to exit")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame_counter += 1

    results = model(frame, conf=0.5)

    stable_text = ""

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            pad = 10

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            crop = frame[y1:y2, x1:x2]

            # OCR setiap 3 frame
            if frame_counter % 3 == 0:

                text, conf = read_plate(crop)

                # simpan hanya hasil bagus
                if conf > 0.6 and len(text) >= 5:
                    ocr_buffer.append(text)

            if len(ocr_buffer) > 0:

                stable_text = Counter(
                    ocr_buffer
                ).most_common(1)[0][0]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.rectangle(
                frame,
                (x1, y1 - 40),
                (x1 + 300, y1),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                stable_text,
                (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    cv2.imshow(
        "ANPR YOLOv8 + EasyOCR",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()