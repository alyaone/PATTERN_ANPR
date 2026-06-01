import pandas as pd
from utils.ocr import read_plate
import cv2
import os

df = pd.read_csv("dataset/plate_text_dataset/label.csv")

correct = 0
total = len(df)

for i, row in df.iterrows():

    img_path = os.path.join(
        "dataset/plate_text_dataset/dataset",
        row["Filename"]
    )

    image = cv2.imread(img_path)

    pred = read_plate(image)

    if pred.replace(" ", "") == str(row["Label"]).replace(" ", ""):
        correct += 1

accuracy = correct / total

print("OCR Accuracy:", accuracy)