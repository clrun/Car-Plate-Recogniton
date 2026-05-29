from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(
        data="datasets/license_plate/plate.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/train_plate"
    )