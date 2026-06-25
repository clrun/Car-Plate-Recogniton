from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# data.yaml 为数据集配置文件，包含类别名称与标注路径
model.train(data="vehicle_types.yaml", epochs=100, imgsz=640)