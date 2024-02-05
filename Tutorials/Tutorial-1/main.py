import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import rate_coding  # Make sure to have rate_coding.py in the same directory
from visualization import visualize_spike_data, visualize_latency_coding
from mnist_setup import convert_to_time

# Training Parameters
batch_size = 128
data_path = '/tmp/data/mnist'

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# MNIST dataset
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# Temporal Dynamics
num_steps = 100

spike_data, targets_it = rate_coding.rate_code_mnist_data(train_loader, num_steps)


# Rate coding experiments
rate_coding.create_rate_coded_vector(10)  # Example with num_steps=10
rate_coding.create_rate_coded_vector(num_steps)  # Example with num_steps=100
rate_coding.rate_code_mnist_data(train_loader, num_steps)


# Visualization and Animation
# Assuming spike_data and targets_it are available from the rate coding part
# visualize_spike_data(spike_data, targets_it, num_steps=100, gain=1.0)  # For original gain
visualize_spike_data(spike_data, targets_it, 100, ffmpeg_path='/usr/bin/ffmpeg', gain=0)
# visualize_spike_data(spike_data, targets_it, num_steps=100)  # Visualize with default settings
# visualize_spike_data(spike_data, targets_it, num_steps=100, gain=0.25)  # For reduced gain

# spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

# Example code snippet for applying latency coding to MNIST data
for data, target in train_loader:
    spike_times = convert_to_time(data, tau=5, threshold=0.01)
   
visualize_latency_coding()

