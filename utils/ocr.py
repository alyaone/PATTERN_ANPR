import cv2
import easyocr
import re

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

def preprocess_plate(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return gray


def read_plate(crop):

    processed = preprocess_plate(crop)

    results = reader.readtext(
        processed,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        paragraph=False
    )

    if len(results) == 0:
        return "", 0.0

    print("OCR RAW:", results)

    # gabungkan semua potongan teks
    text = ""

    # urut kiri ke kanan
    results = sorted(
        results,
        key=lambda x: min([p[0] for p in x[0]])
    )

    for r in results:
        text += r[1]

    text = text.upper()

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text
    )

    confidence = max([r[2] for r in results])

    print(
        f"[OCR] {text} | Confidence: {confidence:.3f}"
    )

    return text, confidence