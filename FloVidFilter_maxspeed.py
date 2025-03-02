import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings
import cv2
import argparse
import os
from datetime import datetime
import logging
import threading
import time
import queue
import json

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.layers is deprecated, please import via timm.layers")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def process_frame(frame, model, processor, device):
    """
    Process a single frame to detect objects (without drawing bounding boxes).
    """
    logging.debug("Starting process_frame")

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    logging.debug("Converted frame to PIL image")

    # Define the object detection task
    task = "<OD>"

    # Process the image and task
    inputs = processor(text=task, images=pil_image, return_tensors="pt").to(device)
    logging.debug("Processed image and task")

    # Generate object detection results
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,  # Reduce the number of generated tokens
        num_beams=3,         # Reduce the number of beams
        do_sample=False,     # Disable sampling
        # Remove or unset temperature, top_k, and top_p
    )
    logging.debug("Generated object detection results")

    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    logging.debug(f"Decoded generated text: {generated_text}")

    # Post-process the generated text to get bounding boxes and labels
    parsed_answer = processor.post_process_generation(generated_text, task=task, image_size=(pil_image.width, pil_image.height))
    logging.debug(f"Post-processed generation: {parsed_answer}")

    # Extract labels and number of objects
    num_objects = 0
    labels = []
    if isinstance(parsed_answer, dict) and '<OD>' in parsed_answer:
        od_data = parsed_answer['<OD>']
        labels = od_data.get('labels', [f"Object {i+1}" for i in range(len(od_data['bboxes']))])
        num_objects = len(labels)
        logging.debug(f"Detected {num_objects} objects with labels: {labels}")

    logging.debug("Ending process_frame")
    return frame, num_objects, labels

# Set the frame skip interval
frame_skip = 10  # Process every 5 frames

def main(video_path):
    logging.debug("Starting main")

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"Using device: {device}")

    # Load the model and processor
    try:
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        logging.debug("Model and processor loaded successfully")
    except OSError as e:
        print(f"Error loading the model: {e}")
        logging.error(f"Error loading the model: {e}")
        return

    # Load the video
    cap = cv2.VideoCapture(video_path)
    logging.debug(f"Loading video from: {video_path}")

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        logging.error(f"Error opening video file: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.debug(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total Frames: {total_frames}")

    # Generate a timestamped output video file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path_base = f"{video_base_name}_filtered_{timestamp}"
    output_path = f"{output_path_base}.mp4"  # Initial output path
    logging.debug(f"Output video base path: {output_path_base}")

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize frame counter and saved frame counter
    frame_counter = 0
    saved_frame_counter = 0
    part_counter = 1
    frames_per_part = total_frames // 1  # Process in 10% chunks

    # List of specified labels to exclude
    exclude_labels = ["human head", "man", "person", "human face", "woman", "helmet", "hat", "footwear", "human hand","human mouth", "human eye", "human eyes", 'jacket', 'hand', 'sneaker', 'sneakers']
    logging.debug(f"Excluding labels: {exclude_labels}")

    # Queue to hold saved frames
    saved_frames_queue = queue.Queue()

    # List to store labels for saved frames
    saved_frames_labels = []

    # Flag to signal the display thread to stop
    stop_display_thread = threading.Event()

    # Variable to track if we are in single-frame processing mode
    single_frame_processing = False
    single_frame_counter = 0
    last_saved_frame = 0

    def display_saved_frames(frame_queue, stop_event):
        """
        Displays saved frames in a loop.
        """
        logging.debug("Starting display_saved_frames thread")
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)  # Wait for 1 second for a frame
                # Resize the frame to 25% of its original size
                resized_frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imshow('Filtered Video - Playback Loop', resized_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):  # Adjust waitKey for desired FPS
                    stop_event.set()
                    break
            except queue.Empty:
                # Queue is empty, but continue looping if the processing thread is still running
                if stop_event.is_set():
                    break
                continue
        cv2.destroyWindow('Filtered Video - Playback Loop')  # Destroy the window after finishing
        logging.debug("Ending display_saved_frames thread")

    # Start the display thread
    display_thread = threading.Thread(target=display_saved_frames, args=(saved_frames_queue, stop_display_thread))
    display_thread.start()

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.debug("End of video stream")
            break

        # Increment frame counter
        frame_counter += 1

        if single_frame_processing:
            # Process each frame individually
            logging.debug(f"Single frame processing: Frame {frame_counter}")
            processed_frame, num_objects, labels = process_frame(frame, model, processor, device)

            if num_objects == 0 or not any(label.lower() in exclude_labels for label in labels):
                # Save the frame
                out.write(processed_frame)
                saved_frame_counter += 1
                logging.debug(f"Frame {frame_counter} saved to output video")
                saved_frames_queue.put(processed_frame)
                saved_frames_labels.append({"frame_number": frame_counter, "labels": labels})
            else:
                logging.debug(f"Frame {frame_counter} excluded due to detected labels.")

            single_frame_counter += 1
            if single_frame_counter >= frame_skip:
                single_frame_processing = False
                single_frame_counter = 0
                logging.debug("Returning to frame skip processing mode.")

        elif frame_counter % frame_skip == 0:
            # Process every frame_skip frames
            logging.debug(f"Frame skip processing: Frame {frame_counter}")
            processed_frame, num_objects, labels = process_frame(frame, model, processor, device)

            if num_objects == 0 or not any(label.lower() in exclude_labels for label in labels):
                # Enter single-frame processing mode for the previous N frames
                single_frame_processing = True
                single_frame_counter = 0
                last_saved_frame = frame_counter
                # Rewind the video to the first of the previous N frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_counter - frame_skip))
                logging.debug(f"Entering single frame processing mode. Rewinding to frame {max(0, frame_counter - frame_skip)}")

                # Need to re-read the first frame
                ret, frame = cap.read()
                frame_counter = max(0, frame_counter - frame_skip) + 1

            else:
                # Exclude the previous N frames by skipping them
                logging.debug(f"Excluding frames {frame_counter - frame_skip + 1} to {frame_counter} due to detected labels.")

        # Display progression ratio
        progression_ratio = (frame_counter / total_frames) * 100
        print(f"Progression: {progression_ratio:.2f}% | Saved Frames: {saved_frame_counter}")

        # Check if it's time to save this part of the video
        if frame_counter % frames_per_part == 0:
            # Release the current video writer
            out.release()
            logging.debug(f"Released video writer for part {part_counter}")

            # Save the labels to a JSON file for this part
            labels_output_path = f"{output_path_base}_part_{part_counter}_labels.json"
            with open(labels_output_path, 'w') as f:
                json.dump(saved_frames_labels, f, indent=4)
            logging.debug(f"Labels saved to: {labels_output_path}")

            # Clear the saved labels for the next part
            saved_frames_labels = []

            # Update the output path for the next part
            output_path = f"{output_path_base}_part_{part_counter}.mp4"
            logging.debug(f"New output video path: {output_path}")

            # Create a new video writer for the next part
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            logging.debug(f"Created video writer for part {part_counter}")

            # Increment the part counter
            part_counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.debug("User interrupted the process")
            stop_display_thread.set()
            break

    # Signal the display thread to stop and wait for it to finish
    stop_display_thread.set()
    display_thread.join()

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.debug("Released all resources")

    print(f"Filtered video saved as: {output_path}")
    logging.debug("Ending main")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a video file for object detection.")
    parser.add_argument("video_path", type=str, help="Path to the video file")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the video path
    main(args.video_path)