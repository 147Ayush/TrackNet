import time
import cv2

def benchmark(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        model.detect_objects(frame)
        frame_count += 1

    end_time = time.time()

    if frame_count == 0:
        print("No frames were processed. Check the video file.")
        return

    fps = frame_count / (end_time - start_time)
    print(f"Inference FPS: {fps}")
    cap.release()
