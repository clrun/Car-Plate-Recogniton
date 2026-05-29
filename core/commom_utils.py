import cv2
import numpy as np

def letterbox(img, new_shape=(640,640)):
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = new_shape[1]-unpad[0], new_shape[0]-unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img

def save_result(img, path):
    cv2.imwrite(path, img)