import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
eye = [ "left", "right"]
rawfilepath = data_dir = os.path.join(os.getcwd(), "EV_Eye_dataset/raw_data")
processedpath = os.path.join(os.getcwd(), "EV_Eye_dataset/processed_data")


# Sensor size
img_size = (346, 240)
num_events = 10000


def display_frames_as_video(frames, frame_size=(640, 480), frame_rate=30, output_file="output_video.mp4"):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    # Write frames to the video file
    for frame in frames:
        frame_resized = cv2.resize(frame, frame_size)
        out.write(frame_resized)

    # Release the VideoWriter object
    out.release()

    # Print a message
    print(f"Video saved as {output_file}")

def visualize_2(timestamp, x, y, pol, img_size):
    img = np.zeros(img_size, np.float32)
    num_timestamp = len(timestamp)
    print("Space-time plot and movie: numevents = ", num_timestamp)
    num_bins = 5
    print("Number of time bins = ", num_bins)

    t_max = np.amax(timestamp)
    t_min = np.amin(timestamp)
    t_range = t_max - t_min
    dt_bin = t_range / num_bins  # size of the time bins (bins)
    t_edges = np.linspace(t_min, t_max, num_bins + 1)  # Boundaries of the bins

    # Compute 3D histogram of events manually with a loop
    # ("Zero-th order or nearest neighbor voting")
    hist3d = np.zeros(img_size + (num_bins,), int)
    for i in range(num_timestamp):
        # Convert x[i] and y[i] to integers
        x_coord = int(x[i])
        y_coord = int(y[i])
        # Check if coordinates are within bounds
        if 0 <= y_coord < img_size[0] and 0 <= x_coord < img_size[1]:
            idx_t = int((timestamp[i] - t_min) / dt_bin)
            # Ensure that idx_t is within the range of valid indices
            if 0 <= idx_t < num_bins:
                hist3d[y_coord, x_coord, idx_t] += 1

    # Plot of the 3D histogram
    fig = plt.figure()
    fig.suptitle('3D histogram (voxel grid), zero-th order voting')
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(hist3d)
    ax.set(xlabel='x', ylabel='time bin', zlabel='y')
    ax.view_init(azim=-90, elev=-180)
    plt.show()
def visualize_1(timestamp, x, y, pol, img_size):
    t_ref = timestamp[-1]  # time of the last event in the packet
    tau = 0.03  # decay parameter (in seconds)
    # Time surface (or time map, or SAE), separated by polarity
    sae_pos = np.zeros(img_size, np.float32)
    sae_neg = np.zeros(img_size, np.float32)
    num_events = len(timestamp)

    # Average timestamp per pixel
    sae = np.zeros(img_size, np.float32)
    count = np.zeros(img_size, int)
    for i in range(num_events):
        # Check if coordinates are within bounds
        if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
            sae[int(y[i]), int(x[i])] += timestamp[i]
            count[int(y[i]), int(x[i])] += 1

    # Compute per-pixel average if count at the pixel is >1
    count[count < 1] = 1  # to avoid division by zero
    sae = sae / count

    fig = plt.figure()
    fig.suptitle('Average timestamps regardless of polarity')
    plt.imshow(sae)
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()


    # Average timestamp per pixel. Separate by polarity
    sae_pos = np.zeros(img_size, np.float32)
    sae_neg = np.zeros(img_size, np.float32)
    count_pos = np.zeros(img_size, int)
    count_neg = np.zeros(img_size, int)
    for i in range(num_events):
        # Check if coordinates are within bounds
        if 0 <= y[i] < img_size[0] and 0 <= x[i] < img_size[1]:
            if pol[i] > 0:
                sae_pos[int(y[i]), int(x[i])] += timestamp[i]
                count_pos[int(y[i]), int(x[i])] += 1
            else:
                sae_neg[int(y[i]), int(x[i])] += timestamp[i]
                count_neg[int(y[i]), int(x[i])] += 1
    # Compute per-pixel average if count at the pixel is >1
    count_pos[count_pos < 1] = 1
    sae_pos = sae_pos / count_pos

    count_neg[count_neg < 1] = 1
    sae_neg = sae_neg / count_neg

    fig = plt.figure()
    fig.suptitle('Average timestamps of positive events')
    plt.imshow(sae_pos)
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()


    fig = plt.figure()
    fig.suptitle('Average timestamps of negative events')
    plt.imshow(sae_neg)
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.show()

def integrate_events_to_frames(events_data, T):
    N = len(events_data)  # Total number of events
    width, height = 346, 240  # Resolution of the Davis camera

    frame_shape = (2, width, height)  # Shape of each frame (2 channels for polarity)

    frames = np.zeros((T,) + frame_shape, dtype=np.float32)  # Initialize frames

    for j in range(T):  # Iterate over frames
        j_l = (N // T) * j  # Start index of events for frame j
        j_r = min((N // T) * (j + 1), N)  # End index of events for frame j

        for p in range(2):  # Iterate over polarity channels
            for x in range(width):  # Iterate over x coordinates
                for y in range(height):  # Iterate over y coordinates
                    # Sum polarity values of events within the range [j_l, j_r)
                    polarity_sum = np.sum(events_data[j_l:j_r, 3] == p)  # Count events with polarity p
                    frames[j, p, x, y] = polarity_sum

    return frames

# Iterate over the desired range
for user_num in range(1, 49):  # (user_num = 1:48)
    for whicheye in eye:
        for session in range(1, 2):  # (session = 1:2)
            for pattern in range(1, 2):  # (pattern = 1:2)
                # Define the directory where the file is located
                load_dir = os.path.join(processedpath, f'Davis_tobii/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')

                # Define the filename
                filename = os.path.join(load_dir, 'data.npy')

                # Load the NumPy array from the file
                data = np.load(filename)

                # Now you can use final_data_loaded as your loaded NumPy array
                print(data.shape)
                timestamp = data[:, 0]
                x = data[:, 1]
                y = data[:, 2]
                p = data[:, 3]
                x_label = data[:, 4]
                y_label = data[:, 5]
                visualize_1(timestamp, x, y, p, img_size)
                # Integrate events into frames
                # T = 24  # Number of frames
                # frames = integrate_events_to_frames(data, T)

                # # Visualize the frames
                # for frame in frames:
                #     count = count + 1
                #     display_frames_as_video(frames, frame_size=(640, 480), frame_rate=T, output_file=f"user{user_num}/{whicheye}/session_{session}_0_{pattern}.mp4")