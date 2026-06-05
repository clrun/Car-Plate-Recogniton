import cv2
import numpy as np
import torch
from ultralytics import YOLO
# 暂时先不导入字符识别相关内容，避免报错
# from core.plate_char_rec import PlateCharNet, rec_plate_char, transform
import os  # 补上os模块导入

# ======================== 请修改以下路径为你自己的 ========================
# 1. 你训练好的车牌检测模型路径（从runs/train_plate/weights/里找best.pt）
PLATE_DETECT_MODEL_PATH = r"D:\Car-Plate-Recogniton\runs\detect\runs\train_plate\train-6\weights\best.pt"
# 2. 测试图片路径（换成你自己的图片）
TEST_IMAGE_PATH = r"D:\Car-Plate-Recogniton\datasets\license_plate\CCPD5000\02-88_85-149&620_400&701-400&689_162&703_159&622_397&608-0_0_17_9_29_27_32-86-34.jpg"
# 3. 结果保存路径
RESULT_SAVE_PATH = r"D:\Car-Plate-Recognition\result\plate_rec_result.jpg"
# ======================================================================

# 加载模型
print("正在加载模型...")
# 加载车牌检测模型
plate_detector = YOLO(PLATE_DETECT_MODEL_PATH)
print("模型加载完成！")

def detect_plate(image_path, save_path):
    """
    车牌检测（先不做字符识别，保证能跑通）
    :param image_path: 输入图片路径
    :param save_path: 结果保存路径
    :return: 带标注的图片
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    img_copy = img.copy()

    # 1. 车牌检测
    print("正在检测车牌...")
    results = plate_detector.predict(img, conf=0.5, verbose=False)
    for res in results:
        boxes = res.boxes
        for box in boxes:
            # 提取车牌框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # 绘制车牌框和置信度
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, f"conf: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_copy)
    print(f"结果已保存到：{save_path}")
    return img_copy

if __name__ == "__main__":
    # 运行测试
    detect_plate(TEST_IMAGE_PATH, RESULT_SAVE_PATH)
    # 显示结果
    result_img = cv2.imread(RESULT_SAVE_PATH)
    if result_img is not None:
        cv2.imshow("车牌检测结果", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()