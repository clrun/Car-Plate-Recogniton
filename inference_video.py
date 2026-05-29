import cv2
from inference_img import CAR_MODEL, PLATE_MODEL, CHAR_MODEL, rec_plate_char, transform

def video_detect(video_path=0):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_copy = frame.copy()

        # 车型检测
        car_res = CAR_MODEL.predict(frame, conf=0.5)
        for r in car_res:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                name = r.names[int(box.cls[0])]
                cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame_copy,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # 车牌检测
        plate_res = PLATE_MODEL.predict(frame, conf=0.4)
        for r in plate_res:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame_copy,(x1,y1),(x2,y2),(255,0,0),2)

        cv2.imshow("Car&Plate Detect", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 0=本地摄像头，填视频路径即为视频检测
    video_detect(0)