# import numpy as np
# import cv2
# import os
# import json
# from datetime import datetime, timedelta
# import bisect
# import matplotlib.pyplot as plt
# eye = [ "left", "right"]
# rawfilepath = data_dir = os.path.join(os.getcwd(), "EV_Eye_dataset/raw_data")
# processedpath = os.path.join(os.getcwd(), "EV_Eye_dataset/processed_data")

# def convert_microseconds_to_timestamp(microseconds):
#     # Convert microseconds to seconds
#     timestamp_seconds = microseconds / 1e6
    
#     # Convert to a datetime object
#     timestamp_datetime = datetime.utcfromtimestamp(timestamp_seconds)
    
#     # Add 7 hours to the datetime object
#     adjusted_datetime = timestamp_datetime + timedelta(hours=7)
    
#     # Convert the adjusted datetime back to a string in the desired format
#     adjusted_timestamp = adjusted_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    
#     return adjusted_timestamp

# def convert_davis_timestamp(tobii_created_time):
#     # Convert the timestamp to a datetime object
#     timestamp_datetime = datetime.strptime(tobii_created_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    
#     # Add 7 hours to the datetime object
#     adjusted_datetime = timestamp_datetime + timedelta(hours=7)
    
#     # Convert the datetime object into a Unix timestamp (seconds since the epoch)
#     unix_timestamp = adjusted_datetime.timestamp()
    
#     # Convert the Unix timestamp to microseconds
#     microseconds_timestamp = int(unix_timestamp * 1e6)
    
#     return microseconds_timestamp
# def read_tobii_data(user_num, session, pattern):
#     # time_raw, d2x, d2y, d3x, d3y, d3z = np.loadtxt(os.path.join(rawfilepath, f'Data_tobii/user{user_num}/session_{session}_0_{pattern}/gazedata.txt'), unpack=True)
#     data_file = os.path.join(rawfilepath, f'Data_tobii/user{user_num}/session_{session}_0_{pattern}/gazedata.txt')

#     # Initialize empty lists to store data
#     timestamps = []
#     d2x_values = []
#     d2y_values = []
#     d3x_values = []
#     d3y_values = []
#     d3z_values = []

#     # Read the file line by line
#     with open(data_file, 'r') as file:
#         for line in file:
#             # Split the line by whitespace and strip any leading/trailing whitespace
#             values = line.strip().split()

#             # If there are no values in the line, append 0 to all lists
#             if len(values) == 0:
#                 timestamps.append(0)
#                 d2x_values.append(0)
#                 d2y_values.append(0)
#                 d3x_values.append(0)
#                 d3y_values.append(0)
#                 d3z_values.append(0)
#             # If there are values in the line, append them to the corresponding lists
#             else:
#                 timestamps.append(float(values[0]))
#                 d2x_values.append(float(values[1]) if len(values) > 1 else 0)
#                 d2y_values.append(float(values[2]) if len(values) > 2 else 0)
#                 d3x_values.append(float(values[3]) if len(values) > 3 else 0)
#                 d3y_values.append(float(values[4]) if len(values) > 4 else 0)
#                 d3z_values.append(float(values[5]) if len(values) > 5 else 0)

#     # Convert lists to numpy arrays
#     time_raw = np.array(timestamps)
#     d2x = np.array(d2x_values)
#     d2y = np.array(d2y_values)
#     d3x = np.array(d3x_values)
#     d3y = np.array(d3y_values)
#     d3z = np.array(d3z_values)
#     return time_raw, d2x, d2y, d3x, d3y, d3z


# def adjust_timestamps(tobii_created_time, tobii_time):
    
#     # Convert tobii_time (seconds) to microseconds and add to created_microseconds
#     adjusted_timestamps = tobii_created_time + np.array(tobii_time) * 1e6
    
#     return adjusted_timestamps


# for user_num in range(1, 49):  # (user_num = 1:48)
#     for whicheye in eye:
#         for session in range(1, 2):  # (session = 1:2)
#             for pattern in range(1, 2):  # (pattern = 1:2)
#                 # DVS frame & events read
#                 davis_path_folder = os.path.join(rawfilepath, f'Data_davis/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')
#                 tobii_path_folder = os.path.join(rawfilepath, f'Data_tobii/user{user_num}/session_{session}_0_{pattern}/')            
#                 event_timestamp, event_x, event_y, event_polarity = np.loadtxt(os.path.join(davis_path_folder, 'events/events.txt'), unpack=True)
#                 # event_timestamp = event_timestamp[:1000]
#                 tobii_time, d2x, d2y, d3x, d3y, d3z = read_tobii_data(user_num, session, pattern)
#                 event_start_timestamp = event_timestamp[0]

#                 # Read the .g3 file
#                 with open(os.path.join(tobii_path_folder,'recording.g3'), "r", encoding='utf-8') as file:
#                     data = json.load(file)
#                 tobii_created_time = convert_davis_timestamp(data["created"])
#                 tobii_timestamp = adjust_timestamps(tobii_created_time, tobii_time)
#                 time_differences = tobii_timestamp - event_timestamp[0]

#                 tobii_timestamp = np.delete(tobii_timestamp, np.where(time_differences < 0))

#                 indices_closest_to_event = []

#                 for event_ts in event_timestamp:
#                     closest_index = bisect.bisect_left(tobii_timestamp, event_ts)
#                     if closest_index == 0:
#                         indices_closest_to_event.append(closest_index)
#                     elif closest_index == len(tobii_timestamp):
#                         indices_closest_to_event.append(closest_index - 1)
#                     else:
#                         left_diff = abs(event_ts - tobii_timestamp[closest_index - 1])
#                         right_diff = abs(event_ts - tobii_timestamp[closest_index])
#                         if left_diff < right_diff:
#                             indices_closest_to_event.append(closest_index - 1)
#                         else:
#                             indices_closest_to_event.append(closest_index)

#                 final_data = []

#                 for idx in indices_closest_to_event:
#                     # Extract data corresponding to the current index
#                     timestamp = tobii_timestamp[idx]
#                     e_x = event_x[idx]
#                     e_y = event_y[idx]
#                     e_polarity = event_polarity[idx]
#                     d2_x = d2x[idx]
#                     d2_y = d2y[idx]
#                     d3_x = d3x[idx]
#                     d3_y = d3y[idx]
#                     d3_z = d3z[idx]
                    
#                     # Append the extracted data as a list to final_data
#                     final_data.append([timestamp, e_x, e_y, e_polarity, d2_x, d2_y, d3_x, d3_y, d3_z])

#                 # Convert final_data to a NumPy array if needed
#                 final_data = np.array(final_data)

#                 # Create a boolean mask where both d2x and d2y are not zero
#                 mask = np.logical_and(final_data[:, 4] != 0, final_data[:, 5] != 0)

#                 # Apply the mask to filter out the data
#                 final_data = final_data[mask]

#                 # # Define the directory to save the file
#                 # save_dir = os.path.join(processedpath, f'Davis_tobii/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')
#                 # # Ensure the directory exists
#                 # os.makedirs(save_dir, exist_ok=True)

#                 # # Define the filename
#                 # filename = os.path.join(save_dir, 'data.npy')

#                 # # Save the NumPy array as a .npy file
#                 # np.save(filename, final_data)
                
import numpy as np
import cv2
import os
import json
from datetime import datetime, timedelta
import bisect
import matplotlib.pyplot as plt

eye = ["left", "right"]
rawfilepath = data_dir = os.path.join(os.getcwd(), "EV_Eye_dataset/raw_data")
processedpath = os.path.join(os.getcwd(), "EV_Eye_dataset/processed_data")

def convert_microseconds_to_timestamp(microseconds):
    timestamp_seconds = microseconds / 1e6
    timestamp_datetime = datetime.utcfromtimestamp(timestamp_seconds)
    adjusted_datetime = timestamp_datetime + timedelta(hours=7)
    adjusted_timestamp = adjusted_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return adjusted_timestamp

def convert_davis_timestamp(tobii_created_time):
    timestamp_datetime = datetime.strptime(tobii_created_time, "%Y-%m-%dT%H:%M:%S.%fZ")
    adjusted_datetime = timestamp_datetime + timedelta(hours=7)
    unix_timestamp = adjusted_datetime.timestamp()
    microseconds_timestamp = int(unix_timestamp * 1e6)
    return microseconds_timestamp

def read_tobii_data(user_num, session, pattern):
    data_file = os.path.join(rawfilepath, f'Data_tobii/user{user_num}/session_{session}_0_{pattern}/gazedata.txt')
    timestamps, d2x_values, d2y_values, d3x_values, d3y_values, d3z_values = [], [], [], [], [], []

    with open(data_file, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) == 0:
                timestamps.append(0)
                d2x_values.append(0)
                d2y_values.append(0)
                d3x_values.append(0)
                d3y_values.append(0)
                d3z_values.append(0)
            else:
                timestamps.append(float(values[0]))
                d2x_values.append(float(values[1]) if len(values) > 1 else 0)
                d2y_values.append(float(values[2]) if len(values) > 2 else 0)
                d3x_values.append(float(values[3]) if len(values) > 3 else 0)
                d3y_values.append(float(values[4]) if len(values) > 4 else 0)
                d3z_values.append(float(values[5]) if len(values) > 5 else 0)

    time_raw = np.array(timestamps)
    d2x = np.array(d2x_values)
    d2y = np.array(d2y_values)
    d3x = np.array(d3x_values)
    d3y = np.array(d3y_values)
    d3z = np.array(d3z_values)
    return time_raw, d2x, d2y, d3x, d3y, d3z

def adjust_timestamps(tobii_created_time, tobii_time):
    adjusted_timestamps = tobii_created_time + np.array(tobii_time) * 1e6
    return adjusted_timestamps

for user_num in range(1, 49):
    for whicheye in eye:
        for session in range(1, 3):
            for pattern in range(1, 3):
                davis_path_folder = os.path.join(rawfilepath, f'Data_davis/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')
                tobii_path_folder = os.path.join(rawfilepath, f'Data_tobii/user{user_num}/session_{session}_0_{pattern}/')
                event_timestamp, event_x, event_y, event_polarity = np.loadtxt(os.path.join(davis_path_folder, 'events/events.txt'), unpack=True)

                tobii_time, d2x, d2y, d3x, d3y, d3z = read_tobii_data(user_num, session, pattern)
                event_start_timestamp = event_timestamp[0]

                with open(os.path.join(tobii_path_folder, 'recording.g3'), "r", encoding='utf-8') as file:
                    data = json.load(file)
                tobii_created_time = convert_davis_timestamp(data["created"])
                tobii_timestamp = adjust_timestamps(tobii_created_time, tobii_time)
                time_differences = tobii_timestamp - event_timestamp[0]

                tobii_timestamp = np.delete(tobii_timestamp, np.where(time_differences < 0))

                indices_closest_to_event = []

                for event_ts in event_timestamp:
                    closest_index = bisect.bisect_left(tobii_timestamp, event_ts)
                    if closest_index == 0:
                        indices_closest_to_event.append(closest_index)
                    elif closest_index == len(tobii_timestamp):
                        indices_closest_to_event.append(closest_index - 1)
                    else:
                        left_diff = abs(event_ts - tobii_timestamp[closest_index - 1])
                        right_diff = abs(event_ts - tobii_timestamp[closest_index])
                        if left_diff < right_diff:
                            indices_closest_to_event.append(closest_index - 1)
                        else:
                            indices_closest_to_event.append(closest_index)

                final_data = []

                for i, event_ts in enumerate(event_timestamp):
                    closest_idx = indices_closest_to_event[i]
                    timestamp = event_ts  # Use the event timestamp
                    e_x = event_x[i]
                    e_y = event_y[i]
                    e_polarity = event_polarity[i]
                    d2_x = d2x[closest_idx]
                    d2_y = d2y[closest_idx]
                    d3_x = d3x[closest_idx]
                    d3_y = d3y[closest_idx]
                    d3_z = d3z[closest_idx]

                    final_data.append([timestamp, e_x, e_y, e_polarity, d2_x, d2_y, d3_x, d3_y, d3_z])

                final_data = np.array(final_data)

                mask = np.logical_and(final_data[:, 4] != 0, final_data[:, 5] != 0)
                final_data = final_data[mask]

                save_dir = os.path.join(processedpath, f'Davis_tobii/user{user_num}/{whicheye}/session_{session}_0_{pattern}/')
                os.makedirs(save_dir, exist_ok=True)

                filename = os.path.join(save_dir, 'data.npy')
                np.save(filename, final_data)
