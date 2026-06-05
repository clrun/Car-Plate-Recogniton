from ultralytics import YOLO
from ultralytics import settings

# 临时修改Ultralytics的数据集目录，避免路径干扰
settings.update({
    'datasets_dir': 'D:/Car-Plate-Recognition/data'
})

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")
    model.train(
        data="data/data.yaml",  # 或用绝对路径 D:/Car-Plate-Recognition/data/data.yaml
        epochs=80,
        imgsz=640,
        batch=4,
        device=0,
        project="runs/train_plate"
    )