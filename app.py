from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# 模型路径
CAR_MODEL_PATH = r"D:\Car-Plate-Recogniton\runs\detect\runs\train_car\train-6\weights\best.pt"
PLATE_MODEL_PATH = r"D:\Car-Plate-Recogniton\runs\detect\runs\train_plate\train-6\weights\best.pt"

# 加载模型
car_model = YOLO(CAR_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

# 文件夹自动生成
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 接收前端上传图片
    img_file = request.files["file"]
    save_path = os.path.join("uploads", img_file.filename)
    img_file.save(save_path)

    # 读取图片
    img = cv2.imread(save_path)
    draw = img.copy()

    # 车型检测 绿色框
    res_car = car_model.predict(img, conf=0.5, verbose=False)
    for r in res_car:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls_name = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(draw,f"{cls_name} {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # 车牌检测 蓝色框（阈值0.2适配小车牌）
    res_plate = plate_model.predict(img, conf=0.2, verbose=False)
    for r in res_plate:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(draw,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(draw,f"Plate {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    # 保存结果图
    out_path = os.path.join("output", img_file.filename)
    cv2.imwrite(out_path, draw)

    # 返回结果图片给前端
    return send_file(out_path)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)