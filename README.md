# Cyber AI Object Detection

This is a simple real-time object detection web application that uses a webcam and AI model to detect objects in live video.

## About the Project

This project was made as part of a learning activity in computer vision and web development. It uses a pre-trained YOLOv8 model to detect objects from a live camera feed.

The system runs on a web interface using Streamlit, making it easy to use without installing complex software.

## Features

- Live webcam object detection  
- Real-time bounding box display  
- Adjustable confidence level  
- Simple and lightweight interface  
- Runs directly in the browser  

## Technologies Used

- Python  
- Streamlit  
- YOLOv8 (Ultralytics)  
- OpenCV  
- Streamlit WebRTC  

## How It Works

1. The webcam captures live video  
2. Frames are processed using YOLOv8  
3. Objects are detected in real time  
4. Bounding boxes are displayed on screen  


Live Demo

- Streamlit App: https://live-object-detection-awjh7hvmgk4kzvv9moi9rh.streamlit.app/
- GitHub Repository: https://github.com/PrinceLoryLordZamora/live-object-detection
- Documentation Report (Google Docs): https://docs.google.com/document/d/1H7836k9Pw45-757DdrLt3rTwHVVOHTnS/edit?usp=sharing&ouid=111899427387002432381&rtpof=true&sd=true
## Setup Instructions

bash id="setup1"
git clone https://github.com/your-username/project-name.git
cd project-name
pip install -r requirements.txt
streamlit run main.py
