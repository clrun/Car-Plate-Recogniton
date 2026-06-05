import cv2
from ultralytics import YOLO

CAR_WEIGHT = r"runs\detect\runs\train_car\train-6\weights\best.pt"
PLATE_WEIGHT = r"runs\detect\runs\train_plate\train-6\weights\best.pt"
cap = cv2.VideoCapture(r"D:\Car-Plate-Recogniton\QQ20260511-161513-HD.mp4") #0=本地摄像头，填视频路径即可读取mp4

car_model = YOLO(CAR_WEIGHT)
plate_model = YOLO(PLATE_WEIGHT)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    draw = frame.copy()

    # 车型：
    car_pred = car_model.predict(frame, conf=0.5, verbose=False)
    for res in car_pred:
        for box in res.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cls_name = res.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(draw,f"{cls_name} {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # 车牌：
    plate_pred = plate_model.predict(frame, conf=0.2, verbose=False)
    for res in plate_pred:
        for box in res.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(draw,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(draw,f"Plate {conf:.2f}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    cv2.imshow("车型+车牌实时检测【Q退出】", draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()