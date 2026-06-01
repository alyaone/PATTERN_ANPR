from ultralytics import YOLO

model = YOLO("runs/anpr_plate_detection/weights/best.pt")

def detect_plate(image):
    results = model(image)
    return results[0]