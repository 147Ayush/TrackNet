import streamlit as st
import cv2
import json
import os
import urllib.request
from detection import ObjectDetector
from json_formatter import format_to_json
from image_retrieval import crop_and_save
from benchmark import benchmark
from utils import map_subobjects

# Set up the title and description of the app
st.title("Real-Time Object Detection with Sub-Object Hierarchy")
st.write("This app detects objects and sub-objects in real-time video streams using a pre-trained Faster R-CNN model.")

# Option for users to upload their video or use the sample video
video_option = st.radio("Choose a video", ("Upload your own video", "Use sample video"))

if video_option == "Use sample video":
    # URL of the video hosted on GitHub (replace <your_github_url>)
    video_url = r"D:\works\Git_hub\Assignment\data\_video.mp4"

    # Download the video file from the URL
    video_path = "temp_video.mp4"
    urllib.request.urlretrieve(video_url, video_path)
    st.write("Sample video loaded from GitHub.")

elif video_option == "Upload your own video":
    # Allow users to upload their video file
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    if uploaded_video is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

# Initialize object detector
detector = ObjectDetector()


# Function to process the uploaded or sample video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    all_detections = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect_objects(frame)

        # Map sub-objects to their parent objects
        relationships = [{"parent": "person", "child": "helmet"}]  # Example
        detections = map_subobjects(detections, relationships)

        # Save cropped sub-objects images
        output_dir = "data/output/"
        for obj in detections:
            for sub_obj in obj["sub_objects"]:
                save_path = f"{output_dir}frame_{frame_index}_{obj['name']}_{sub_obj['name']}.jpg"
                crop_and_save(frame, sub_obj["bbox"], save_path)

        # Format JSON output and display detections
        all_detections.append(format_to_json(detections))
        frame_index += 1

    # Save JSON output
    with open(f"{output_dir}detections.json", "w") as f:
        json.dump(all_detections, f, indent=4)

    # Return output directory to the user
    return output_dir, all_detections


# Start processing when video is available
if video_path:
    st.write("Processing video...")
    output_dir, detections = process_video(video_path)

    # Show output to the user
    st.write(f"Video processed successfully! The detections and cropped images are saved in the '{output_dir}' folder.")

    # Display the detections JSON
    st.subheader("JSON Output")
    st.json(detections)

    # Show cropped images as an example
    st.subheader("Cropped Images")
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".jpg"):
            st.image(os.path.join(output_dir, file_name))

    # Benchmark performance
    st.subheader("Benchmarking FPS")
    benchmark(detector, video_path)
