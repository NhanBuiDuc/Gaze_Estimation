import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_dir = os.path.join(os.getcwd(), "EV_Eye_dataset/raw_data/Data_davis_labelled_with_mask")
whicheye = "left"  # or "right"

# Function to read data
def read_data(session):
    data_path = os.path.join(data_dir, whicheye, f"{session}.h5")
    with h5py.File(data_path, 'r') as f:
        data = f['data'][:]  # Assuming data is stored under 'data' key
        label = f['label'][:]  # Assuming label is stored under 'label' key
        # Transpose and reshape the data and label
        data = np.transpose(data, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
    return data, label

# Visualize data
def visualize_data(data):
    # Plot or visualize your data here
    plt.imshow(data[-1])  # Display the first sample in the batch
    plt.show()

# Get list of file names
file_names = os.listdir(os.path.join(data_dir, whicheye))

# Example usage
user = 1
for file_name in file_names:
    if file_name.endswith('.h5'):
        session = file_name[:-3]  # Remove .h5 extension
        data, label = read_data(session)
        visualize_data(data)  # Visualize the data
        visualize_data(label)  # Visualize the label
