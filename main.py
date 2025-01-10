import cv2
import json
import threading
import queue
from detection import ObjectDetector
from json_formatter import format_to_json
from image_retrieval import crop_and_save
from benchmark import benchmark
from utils import map_subobjects


frame_queue = queue.Queue(maxsize=10)
detections = []

def video_capture_thread(video_path):
    """Thread that reads frames from the video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        frame_queue.put((frame, frame_index))
        frame_index += 1

    cap.release()

def detection_thread(detector, output_dir):
    """Thread that processes frames and performs object detection."""
    global detections
    while True:
        if not frame_queue.empty():
            frame, frame_index = frame_queue.get()  # Get the next frame from the queue


            detected_objects = detector.detect_objects(frame)


            relationships = [{"parent": "person", "child": "helmet"}]
            detected_objects = map_subobjects(detected_objects, relationships)


            for obj in detected_objects:
                for sub_obj in obj["sub_objects"]:
                    save_path = f"{output_dir}frame_{frame_index}_{obj['name']}_{sub_obj['name']}.jpg"
                    crop_and_save(frame, sub_obj["bbox"], save_path)


            detections.append(format_to_json(detected_objects))


            for obj in detected_objects:
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, obj["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


                for sub_obj in obj["sub_objects"]:
                    x1_s, y1_s, x2_s, y2_s = sub_obj["bbox"]
                    cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 0, 255), 2)
                    cv2.putText(frame, sub_obj["name"], (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


            cv2.imshow('Video', frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    video_path = r"D:\works\Git_hub\Assignment\data\_video.mp4"  # Ensure the path is correct
    output_dir = "data/output/"
    detector = ObjectDetector()


    capture_thread = threading.Thread(target=video_capture_thread, args=(video_path,))
    capture_thread.start()


    detect_thread = threading.Thread(target=detection_thread, args=(detector, output_dir))
    detect_thread.start()


    capture_thread.join()
    detect_thread.join()


    with open(f"{output_dir}detections.json", "w") as f:
        json.dump(detections, f, indent=4)


    benchmark(detector, video_path)


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
