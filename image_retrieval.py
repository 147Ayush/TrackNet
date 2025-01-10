import cv2
import os

def crop_and_save(frame, bbox, save_path):
    x1, y1, x2, y2 = bbox
    cropped_image = frame[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cropped_image)
