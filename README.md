# Smart Surveillance System

## Description
# Intelligent Surveillance System

This project is a real-time intelligent surveillance system developed in Python using Computer Vision.

The system combines motion detection, Region of Interest (ROI) analysis, and YOLOv8-based person detection to efficiently monitor video streams from recorded videos or live cameras. By executing object detection only when significant movement is detected, the system reduces unnecessary processing and improves overall performance.

When a person is detected with sufficient confidence, the system automatically:

* Registers the event in a structured log file.
* Captures and stores evidence images.
* Generates a short video clip containing the moments preceding the detection.
* Provides real-time monitoring through an interactive web dashboard.

The dashboard, built with Streamlit, allows users to:

* Review detection history.
* Browse captured images.
* Download recorded event clips.
* Filter events by date, confidence level, and number of detected people.
* Analyze activity patterns through statistical visualizations.

### Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Streamlit
* Pandas
* Plotly
* PIL
* HTML
* CSS

### Main Features

* Real-time person detection.
* Motion-triggered AI inference for performance optimization.
* Configurable Region of Interest (ROI).
* Adaptive motion thresholding.
* Automatic image capture and event logging.
* Event clip generation.
* Interactive monitoring dashboard.
* Statistical analysis of surveillance activity.

This project was designed as an end-to-end computer vision solution, combining video processing, machine learning, data analysis, and web-based visualization into a single integrated system.
