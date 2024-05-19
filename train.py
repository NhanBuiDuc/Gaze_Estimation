from dataset import EventToFrameDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from net import Net
import torch.nn as nn
# Create the dataset instance
dataset = EventToFrameDataset(user_num=1, eye="left", session=1, pattern=1, window_size="skip", time_window_microseconds=1)

# Split the dataset into training, validation, and test sets
train_indices, valid_indices, test_indices = dataset.split_dataset()

# Create Subset objects for each split
train_dataset = Subset(dataset, train_indices)
valid_dataset = Subset(dataset, valid_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoader instances for each set
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the network, loss function, and optimizer
timesteps = 1000  # Define the number of timesteps
hidden = 4  # Define the number of hidden units
output_size = 2  # Define the output size to match label size

model = Net(timesteps, hidden, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Define the number of epochs

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Data shape needs to match (timesteps, batch_size, features)
        data = data.permute(1, 0, 2)  # Permute to match (timesteps, batch_size, features)
        outputs = model(data)
        
        # The output is of shape (timesteps, batch_size, output_size)
        # We are interested in the final output
        final_outputs = outputs[-1]  # Taking the last timestep output
        
        loss = criterion(final_outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')