from ultralytics import YOLO
import os

def train_vehicle_model():
    # ===================== 模型初始化 =====================
    # 迁移学习：加载官方预训练权重，自定义数据集微调
    # 可选型号：yolov8n/s/m/l/x  精度递增、速度递减
    model = YOLO("yolov8n.pt")

    # ===================== 训练参数配置 =====================
    results = model.train(
        # 基础配置
        data="configs/vehicle_types.yaml",
        epochs=100,
        imgsz=640,
        batch=16,                # 显存不足可调小为8、4
        device=0,                # GPU设备号，CPU运行填 "cpu"
        workers=4,
        seed=42,
        
        # 优化器与学习率
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 数据增强
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        flipud=0.0,
        
        # 保存与日志
        project="runs/vehicle_detect",
        name="exp01",
        save_period=10,
        exist_ok=False,
        verbose=True,
        plots=True
    )

    # ===================== 验证评估 =====================
    metrics = model.val()
    print("=" * 50)
    print(f"验证集 mAP@0.5: {metrics.box.map50:.4f}")
    print(f"验证集 mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"各类别平均精确率: {metrics.box.mp:.4f}")
    print("=" * 50)

    # ===================== 单图测试 =====================
    test_img = "test_car.jpg"
    if os.path.exists(test_img):
        model.predict(
            source=test_img,
            save=True,
            conf=0.5,
            iou=0.45
        )
        print("测试完成，结果保存在 runs/detect/predict 目录下")

if __name__ == "__main__":
    train_vehicle_model()
