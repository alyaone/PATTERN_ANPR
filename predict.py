import cv2
from utils.detector import detect_plate
from utils.ocr import read_plate

image_path = "test.jpg"
image = cv2.imread(image_path)

result = detect_plate(image)

for box in result.boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    crop = image[y1:y2, x1:x2]

    text = read_plate(crop)

    print("PLATE:", text)

    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(image, text, (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("ANPR", image)
cv2.waitKey(0)