from ultralytics import YOLO

if __name__ == "__main__":

    # =========================
    # LOAD PRETRAINED MODEL
    # =========================
    model = YOLO("yolov8n.pt")

    # =========================
    # TRAINING CONFIG
    # =========================
    results = model.train(
        data="data.yaml",

        epochs=200,              # max epoch
        imgsz=640,
        batch=16,

        optimizer="AdamW",
        lr0=0.01,
        cos_lr=True,

        patience=30,             # EARLY STOPPING (jika tidak improve)

        # =========================
        # SAVE CONFIG
        # =========================
        save=True,               # save model
        save_period=1,           # checkpoint tiap epoch (biar aman)

        # =========================
        # MONITORING
        # =========================
        plots=True,              # confusion matrix, PR curve, F1 curve
        verbose=True,            # tampilkan epoch + metrics di terminal

        # =========================
        # OUTPUT FOLDER
        # =========================
        project="runs",
        name="anpr_yolov8_plate"
    )

    # =========================
    # TRAINING DONE
    # =========================
    print("\n============================")
    print("TRAINING SELESAI")
    print("============================\n")

    # =========================
    # LOAD BEST MODEL
    # =========================
    best_model_path = "runs/anpr_yolov8_plate/weights/best.pt"
    model = YOLO(best_model_path)

    # =========================
    # FINAL EVALUATION
    # =========================
    metrics = model.val()

    print("\n===== FINAL METRICS =====")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")
    print(f"mAP@50      : {metrics.box.map50:.4f}")
    print(f"mAP@50-95   : {metrics.box.map:.4f}")

    # =========================
    # SAVE METRICS TO FILE
    # =========================
    import os
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/final_metrics.txt", "w") as f:
        f.write("===== FINAL METRICS =====\n")
        f.write(f"Precision : {metrics.box.mp:.4f}\n")
        f.write(f"Recall    : {metrics.box.mr:.4f}\n")
        f.write(f"mAP@50    : {metrics.box.map50:.4f}\n")
        f.write(f"mAP@50-95 : {metrics.box.map:.4f}\n")

    print("\nMetrics saved to outputs/final_metrics.txt")
    print(f"Best model saved at: {best_model_path}")