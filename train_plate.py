from ultralytics import YOLO
import argparse

def train_plate_bbox():
    """矩形框车牌检测"""
    model = YOLO("yolov8n.pt")
    model.train(
        data="configs/plate_detect.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="runs/plate_detect",
        name="exp_bbox",
        plots=True
    )
    model.val()
    print("矩形车牌检测训练完成")

def train_plate_keypoint():
    """关键点车牌检测（支持倾斜矫正）"""
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data="configs/plate_kpt.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        kobj=1.0,
        project="runs/plate_kpt",
        name="exp_kpt",
        plots=True
    )
    model.val()
    print("关键点车牌检测训练完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="bbox", 
                        choices=["bbox", "kpt"], help="训练模式：bbox矩形框 / kpt关键点")
    args = parser.parse_args()

    if args.mode == "bbox":
        train_plate_bbox()
    else:
        train_plate_keypoint()