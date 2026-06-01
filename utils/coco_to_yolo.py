import json
import os

def convert_coco_to_yolo(coco_json, img_dir, output_label_dir):

    os.makedirs(output_label_dir, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:

        img = images[ann["image_id"]]

        img_w = float(img["width"])
        img_h = float(img["height"])

        x, y, w, h = map(float, ann["bbox"])   

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w /= img_w
        h /= img_h

        label_path = os.path.join(
            output_label_dir,
            img["file_name"].replace(".jpg", ".txt")
        )

        with open(label_path, "a") as f:
            f.write(f"0 {x_center} {y_center} {w} {h}\n")


if __name__ == "__main__":

    convert_coco_to_yolo(
        coco_json=r"C:\Users\Alyapresilis\Documents\Kuliah Elins\SEMESTER 6\Pengenalan Pola\Project_UAS\dataset\plate_detection_dataset\plate_detection_dataset\annotations\annotations.json",
        img_dir=r"C:\Users\Alyapresilis\Documents\Kuliah Elins\SEMESTER 6\Pengenalan Pola\Project_UAS\dataset\plate_detection_dataset\plate_detection_dataset\images",
        output_label_dir=r"C:\Users\Alyapresilis\Documents\Kuliah Elins\SEMESTER 6\Pengenalan Pola\Project_UAS\dataset/labels"
    )