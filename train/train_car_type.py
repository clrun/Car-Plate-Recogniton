from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv8s预训练权重
    model = YOLO("yolov8s.pt")
    # 开始训练，修改为自己数据集yaml路径
    model.train(
        data="datasets/car_type/car.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,  # 0为GPU，cpu用device="cpu"
        patience=20,
        save=True,
        project="runs/train_car"
    )