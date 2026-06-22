# Installation Guide

If you are reading this document, you may be interested in installing, testing, extending, or optimizing this surveillance system.

The following instructions explain how to configure and run the project.

## 1. Clone or Download the Project

Download the complete project and open it in Visual Studio Code.

## 2. Configure the Video Source

You can use either:

* A recorded video file.
* A webcam connected to your computer.
* An external camera.

If you want to analyze a video, place the file inside the project directory and specify its name in the configuration file.

If you want to use a webcam or camera, leave the video path empty or provide an invalid path. The system will automatically attempt to connect to the default camera available on the system.

## 3. Configure the System

Open `configROI.py` and modify the parameters according to your needs.

### Video Source

**VIDEO_PATH**

Name of the video file to be analyzed.


### Region of Interest (ROI)

**ROI_X, ROI_Y**

Coordinates of the upper-left corner of the Region of Interest (ROI).


**ROI_W, ROI_H**

Width and height of the ROI.

The surveillance system will only analyze movement and perform detections inside this region.


### Detection Thresholds


**MIN_CONFIDENCE_COUNT**

Minimum confidence required for a detection to be registered in the Event Log.


**MIN_CONFIDENCE_SAVE**

Minimum confidence required to save an image of the detected event.


**MIN_CONFIDENCE_CLIP**

Minimum confidence required to generate and save an event clip.


### Output Configuration

**OUTPUT_DIR**

Folder where captured images will be stored.


**VIDEO_OUTPUT_DIR**

Folder where generated clips will be stored.


**CSV_PATH**

Path of the CSV file used to store event records.

### Clip Configuration

**SAVE_CLIPS**

Set to `True` to enable automatic clip generation.

Set to `False` to disable clip generation.

**PRE_EVENT_SECONDS**

Number of seconds stored before a detection occurs.

### Cooldowns

**COOLDOWN_DETECTION**

Minimum time between detection log entries.

**COOLDOWN_SAVE**

Minimum time between image captures.

**COOLDOWN_PRINT**

Minimum time between console messages.

## 4. Launch the Dashboard

Open a terminal and run:

```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your web browser.

## 5. Use the Dashboard

The dashboard allows you to:

* Review detected events.
* View saved images.
* Download generated clips.
* Filter events by different criteria.
* Analyze activity statistics and visualizations.

## Learning Path

If you are interested in understanding how the project was developed from scratch, you can follow the development stages described in the main `README.md` file.

These stages document the progression from basic motion detection to the complete intelligent surveillance system presented in this repository.
