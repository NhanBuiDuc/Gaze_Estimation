import numpy as np
import os
from PIL import Image

# Define paths
eye = ["left", "right"]
rawfilepath = os.path.join(os.getcwd(), "EV_Eye_dataset/raw_data")
processedpath = os.path.join(os.getcwd(), "EV_Eye_dataset/processed_data")

# Sensor size
img_size = (346, 240)
screen_size = (1920, 1080)
time_window_microseconds = 64  # 64 microseconds

def accumulate_events_and_labels_to_frames(data, img_size, screen_size, time_window_microseconds):
    event_frames = []
    gaze_frames = []
    start_time = data[0, 0]
    current_time = start_time
    current_event_frame = np.zeros(img_size, dtype=np.uint8)
    current_gaze_frame = np.zeros(screen_size, dtype=np.uint8)
    
    for event in data:
        timestamp, x, y, polarity, label_2d_x, label_2d_y = event[:6]
        x, y = int(x), int(y)
        label_x = int(label_2d_x * screen_size[0])
        label_y = int(label_2d_y * screen_size[1])
        
        if (timestamp - current_time) / 1000.0 < time_window_microseconds:
            if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
                current_event_frame[y, x] = 255  # Mark the pixel
            if 0 <= label_x < screen_size[0] and 0 <= label_y < screen_size[1]:
                current_gaze_frame[label_y-10:label_y+10, label_x-10:label_x+10] = 255  # Draw gaze circle
        else:
            event_frames.append(current_event_frame)
            gaze_frames.append(current_gaze_frame)
            current_event_frame = np.zeros(img_size, dtype=np.uint8)
            current_gaze_frame = np.zeros(screen_size, dtype=np.uint8)
            current_time = timestamp

    if np.any(current_event_frame):
        event_frames.append(current_event_frame)  # Append the last frame if it's not empty
        gaze_frames.append(current_gaze_frame)  # Append the last gaze frame if it's not empty

    return event_frames, gaze_frames

def save_frames_as_gif(frames, output_file="output.gif", frame_duration=10):
    pil_images = [Image.fromarray(frame) for frame in frames]
    pil_images[0].save(output_file, save_all=True, append_images=pil_images[1:], duration=frame_duration, loop=0)
    print(f"GIF saved as {output_file}")

def save_image_frames_as_gif(frames, output_file="output_gaze.gif", frame_duration=10):
    pil_images = [Image.fromarray(frame, mode='L') for frame in frames]
    pil_images[0].save(output_file, save_all=True, append_images=pil_images[1:], duration=frame_duration, loop=0)
    print(f"GIF saved as {output_file}")

# Main processing
for user_num in range(1, 49):
    for whicheye in eye:
        for session in range(1, 3):
            for pattern in range(1, 3):
                load_dir = os.path.join(processedpath, f'Davis_tobii/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')
                filename = os.path.join(load_dir, 'data.npy')

                if os.path.exists(filename):
                    data = np.load(filename)

                    if data.shape[1] >= 6:
                        event_frames, gaze_frames = accumulate_events_and_labels_to_frames(data, img_size, screen_size, time_window_microseconds)
                        output_file = f"user{user_num}_{whicheye}_session{session}_pattern{pattern}.gif"
                        save_frames_as_gif(event_frames, output_file=output_file)

                        output_gaze_file = f"user{user_num}_{whicheye}_session{session}_pattern{pattern}_gaze.gif"
                        save_image_frames_as_gif(gaze_frames, output_file=output_gaze_file)
                    else:
                        print(f"Unexpected data shape {data.shape} in file {filename}")
                else:
                    print(f"File not found: {filename}")
