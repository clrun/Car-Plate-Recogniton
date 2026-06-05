import cv2
from ultralytics import YOLO
import os

# =====================【3个路径】=====================
# 1、车型训练好的best.pt权重
CAR_WEIGHT = r"runs\detect\runs\train_car\train-6\weights\best.pt"
# 2、车牌训练好的best.pt权重
PLATE_WEIGHT = r"runs\detect\runs\train_plate\train-6\weights\best.pt"
# 3、待测试图片路径
TEST_IMG_PATH = r"D:\Car-Plate-Recogniton\datasets\license_plate\CCPD5000\0225-88_92-225&564_466&667-469&643_229&657_233&567_473&553-0_0_22_1_24_30_25-68-38.jpg"
SAVE_RESULT_PATH = r"result/final_out.jpg"
# ==============================================================

# 加载双模型
car_model = YOLO(CAR_WEIGHT)
plate_model = YOLO(PLATE_WEIGHT)

def detect_car_and_plate(img_path, save_path):
    # 读取原图
    img = cv2.imread(img_path)
    draw_img = img.copy()

    # ----------1.车型检测：绿色框+车型名称----------
    car_res = car_model.predict(img, conf=0.5, verbose=False)
    for res in car_res:
        boxes = res.boxes
        cls_names = res.names
        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = cls_names[cls_idx]
            # 画车型框(绿色)
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(draw_img,f"{cls_name} {conf:.2f}",
                        (x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # ----------2.车牌检测：蓝色框+LicensePlate----------
    plate_res = plate_model.predict(img, conf=0.4, verbose=False)
    for res in plate_res:
        boxes = res.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # 画车牌框(蓝色)
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(draw_img,f"Plate {conf:.2f}",
                        (x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    # 保存+弹窗展示
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    cv2.imwrite(save_path,draw_img)
    print(f"✅ 检测完成，结果保存：{save_path}")
    cv2.imshow("车型+车牌检测结果",draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_car_and_plate(TEST_IMG_PATH,SAVE_RESULT_PATH)