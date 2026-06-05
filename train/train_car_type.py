from ultralytics import YOLO
from ultralytics import settings



settings.update({
    'datasets_dir': 'D:/Car-Plate-Recognition/data'
})

if __name__ == "__main__":
    # 加载YOLOv8s预训练权重
    model = YOLO("yolov8s.pt")

    # 开始训练：修改为你自己数据集yaml路径（关键修正！）
    model.train(
        # 路径1：相对路径（推荐，和你的项目结构匹配）
        data="data/vehicle/vehicle.data.yaml",
        
        # 路径2：如果还是报错，用绝对路径（改成你自己的盘符）
        # data="D:/Car-Plate-Recognition/data/vehicle.data.yaml",
        
        epochs=100,
        imgsz=640,
        batch=4,  # 显存不够可以改成2
        device=0,  # 没有GPU就改成 device="cpu"
        patience=20,
        save=True,
        project="runs/train_car"
    )