import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
class EventToFrameDataset(Dataset):
    def __init__(self, user_num, eye, session, pattern, window_size="stack", time_window_microseconds=1000, processed_path=None):
        # Set default path if none provided
        if processed_path is None:
            processed_path = os.path.join(os.getcwd(), "EV_Eye_dataset/processed_data")

        # Sensor size
        self.img_size = (346, 240)
        self.screen_size = (1920, 1080)
        self.window_size = window_size
        self.time_window_microseconds = time_window_microseconds  # in microsecond

        # Define the directory where the file is located
        load_dir = os.path.join(processed_path, f'Davis_tobii/user{user_num}/{eye}/session_{session}_0_{pattern}/')

        # Define the filename
        filename = os.path.join(load_dir, 'data.npy')

        # Load the NumPy array from the file
        self.data = np.load(filename)

        # Prepare the data and labels
        self.train_data, self.labels = self.make_data()
        self.train_data = self.add_padding()
    def make_data(self):
        event_frames = []
        gaze_labels = []

        start_time = self.data[0, 0]
        fixed_size = self.time_window_microseconds * 1000
        end_time = start_time + fixed_size
        event_frame = np.zeros((fixed_size, 4))
        flag_reset = False
        for event in self.data:    
            timestamp, x, y, polarity, label_x, label_y = event[:6]
            if flag_reset:
                start_time = timestamp
                end_time = start_time + fixed_size
            labels = [label_x, label_y]
            if timestamp < end_time:
                flag_reset = False
                # if 0 <= x < self.img_size[1] and 0 <= y < self.img_size[0]:
                index = timestamp - start_time
                event_frame[int(index)] = [timestamp, x, y, polarity]
                last_labels = [label_x, label_y]
                if timestamp == end_time:
                    gaze_labels.append(labels)
            else:
                gaze_labels.append(last_labels)
                # Pad current_frame to the fixed size (e.g., 1000 events)
                event_frames.append(event_frame)
                event_frame = np.zeros((fixed_size, 4))
                flag_reset = True

        return event_frames, gaze_labels

    def add_padding(self):
        max_length = max(len(frame) for frame in self.train_data)
        empty_event = [0, 0, 0, 0]
        padded_data = []
        
        for frame in self.train_data:
            padded_frame = frame.tolist()  # Convert to list for easy appending
            while len(padded_frame) < max_length:
                padded_frame.append(empty_event)
            padded_data.append(np.array(padded_frame))
        
        return padded_data

    def split_dataset(self, train_ratio=0.6, valid_ratio=0.3, test_ratio=0.1):
        train_size = int(train_ratio * len(self))
        valid_size = int(valid_ratio * len(self))
        test_size = len(self) - train_size - valid_size

        indices = np.arange(len(self))
        train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=42)
        valid_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=42)
        
        return train_indices, valid_indices, test_indices
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        train_data_tensor = torch.FloatTensor(self.train_data[idx])
        labels_tensor = torch.FloatTensor(self.labels[idx])
        return train_data_tensor, labels_tensor
