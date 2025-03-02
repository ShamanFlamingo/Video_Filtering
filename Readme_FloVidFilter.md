# FloVidFilter: Video Filtering with Florence-2 Object Detection

## 1. Introduction

This document describes the `FloVidFilter_maxspeed.py` script, a Python application designed to filter video content based on object detection. The script utilizes the Florence-2 model from Microsoft to identify objects within video frames and exclude segments containing specified objects. The script is optimized for speed and efficiency, leveraging GPU acceleration when available.

## 2. Requirements

### 2.1. Functional Requirements

*   **Video Input:** The script must accept a video file as input.
*   **Object Detection:** The script must use the Florence-2 model to detect objects in each frame.
*   **Object Exclusion:** The script must allow users to specify a list of object labels to exclude.
*   **Video Filtering:** The script must filter out video segments containing any of the excluded objects.
*   **Video Output:** The script must produce a filtered video file as output.
*   **Performance:** The script should process video frames efficiently, utilizing GPU acceleration when available.
*   **Progress Monitoring:** The script should display a progress indicator during processing.
*   **Segmented Output:** The script should divide the output video into smaller parts to prevent large file sizes.
*   **Label Saving:** The script should save the labels of saved frames to a JSON file for each part.
*   **Display Filtered Video:** The script should display the filtered video in a separate window.

### 2.2. Non-Functional Requirements

*   **Accuracy:** The object detection should be reasonably accurate.
*   **Efficiency:** The script should process videos in a timely manner.
*   **Usability:** The script should be easy to use with clear instructions.
*   **Maintainability:** The code should be well-structured and documented for easy maintenance.
*   **Logging:** The script should log important events and errors for debugging.

### 2.3. Dependencies

*   Python 3.6+
*   PyTorch
*   OpenCV (`cv2`)
*   Transformers library (`transformers`)
*   PIL (Pillow)
*   `argparse`
*   `datetime`
*   `logging`
*   `threading`
*   `time`
*   `queue`
*   `json`

## 3. Design

### 3.1. Architecture

The script follows a modular design, with distinct functions for:

*   Loading the model and processor
*   Processing individual frames
*   Managing video input and output
*   Displaying the filtered video

### 3.2. Components

*   **`main(video_path)`:** The main function that orchestrates the video processing pipeline.
*   **`process_frame(frame, model, processor, device)`:** Detects objects in a single frame using the Florence-2 model.
*   **`display_saved_frames(frame_queue, stop_event)`:** Displays saved frames in a loop in a separate thread.

### 3.3. Data Flow

1.  The script receives the video path as a command-line argument.
2.  The `main` function loads the Florence-2 model and processor.
3.  The script opens the video file using OpenCV.
4.  The script iterates through each frame of the video:
    *   The `process_frame` function detects objects in the frame.
    *   If the frame does not contain any of the excluded objects, it is saved to the output video.
    *   The script displays a progress indicator.
    *   The script divides the output video into smaller parts based on `frames_per_part`.
    *   The script saves the labels of saved frames to a JSON file for each part.
5.  The script releases all resources and displays a completion message.

## 4. Implementation Details

### 4.1. Code Structure

The code is organized into the following sections:

1.  **Import Statements:** Imports necessary libraries.
2.  **Configuration:** Defines configurable parameters such as `frame_skip`, `exclude_labels`, and `frames_per_part_percentage`.
3.  **`process_frame` Function:** Processes a single frame to detect objects.
4.  **`main` Function:** Orchestrates the video processing pipeline.
5.  **`display_saved_frames` Function:** Displays saved frames in a loop in a separate thread.
6.  **Main Execution Block:** Parses command-line arguments and calls the `main` function.

### 4.2. Key Algorithms

*   **Object Detection:** The script uses the Florence-2 model for object detection. The `process_frame` function converts the OpenCV frame to a PIL image, preprocesses it using the Florence-2 processor, and then uses the model to generate object detection results.
*   **Video Filtering:** The script filters video segments based on the presence of excluded objects. If a frame contains any of the excluded objects, it is not saved to the output video.
*   **Segmented Output:** The script divides the output video into smaller parts based on the `frames_per_part` variable. This prevents the output video file from becoming too large.

### 4.3. Configuration Parameters

The following configuration parameters can be adjusted to modify the script's behavior:

*   **`frame_skip`:** The number of frames to skip between processing frames.
*   **`exclude_labels`:** A list of object labels to exclude from the output video.
*   **`frames_per_part_percentage`:** The percentage of total frames to include in each output video part.

## 5. Usage

### 5.1. Command-Line Arguments

The script accepts the following command-line argument:

*   `video_path`: The path to the input video file.

### 5.2. Example Usage

```bash
python FloVidFilter_maxspeed.py input.mp4
```

## 6. Error Handling

The script includes error handling for the following scenarios:

*   Failure to load the Florence-2 model
*   Failure to open the input video file

## 7. Future Enhancements

*   **GUI:** Develop a graphical user interface (GUI) for easier configuration and usage.
*   **More Sophisticated Filtering:** Implement more sophisticated filtering options, such as filtering based on object size or location.
*   **Real-time Processing:** Implement real-time video processing capabilities.
*   **Configuration File:** Load configuration parameters from a file (e.g., JSON or YAML) instead of hardcoding them in the script.
*   **Progress Bar:** Implement a more visual progress bar.

## 8. Notes

*   Ensure that all required dependencies are installed before running the script.
*   The script requires a significant amount of memory, especially when using GPU acceleration.
*   The processing time depends on the video resolution, frame rate, and the complexity of the scene.