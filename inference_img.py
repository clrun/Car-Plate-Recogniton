from ultralytics import YOLO
import cv2
import torch
from core.common_utils import save_result
from core.plate_char_rec import PlateCharNet, rec_plate_char, transform

# 加载双模型
CAR_MODEL = YOLO("weights/best_car.pt")    # 车型检测权重
PLATE_MODEL = YOLO("weights/best_plate.pt") # 车牌检测权重
CHAR_MODEL = PlateCharNet()
# 加载训练好的字符识别权重
CHAR_MODEL.load_state_dict(torch.load("weights/plate_char.pth",map_location="cpu"))

def car_plate_detect(img_path, save_path):
    img = cv2.imread(img_path)
    img_copy = img.copy()

    # 1. 车型检测
    car_results = CAR_MODEL.predict(img, conf=0.5)
    for res in car_results:
        boxes = res.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = res.names[cls_id]
            # 绘制车型框
            cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img_copy,f"{cls_name} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # 2. 车牌检测+字符识别
    plate_results = PLATE_MODEL.predict(img, conf=0.4)
    for res in plate_results:
        boxes = res.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            # 截取车牌区域
            plate_img = img[y1:y2,x1:x2]
            # 车牌字符识别（简易逐字符分割识别）
            plate_tensor = transform(plate_img)
            plate_text = rec_plate_char(plate_tensor, CHAR_MODEL)
            # 绘制车牌
            cv2.rectangle(img_copy,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(img_copy,f"Plate:{plate_text}",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    save_result(img_copy, save_path)
    print(f"检测完成，结果保存至：{save_path}")
    return img_copy

if __name__ == "__main__":
    car_plate_detect("test.jpg","result/test_out.jpg")