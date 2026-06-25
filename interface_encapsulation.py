from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.post("/api/detect")
async def detect_api(image: UploadFile = File(...)):
    # 读取上传的图片
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 调用检测函数
    vehicle_res = detect_vehicles(img)
    plate_res = detect_plates(img)

    # 格式化返回（与前端约定格式一致）
    return {
        "vehicles": [
            {
                "x1": v["bbox"][0], "y1": v["bbox"][1],
                "x2": v["bbox"][2], "y2": v["bbox"][3],
                "type": v["name"], "confidence": v["confidence"]
            } for v in vehicle_res
        ],
        "plates": [
            {
                "x1": p["bbox"][0], "y1": p["bbox"][1],
                "x2": p["bbox"][2], "y2": p["bbox"][3],
                "number": "", "confidence": p["confidence"]
            } for p in plate_res
        ]
    }