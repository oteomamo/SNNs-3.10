# 1.1. IMPORT PACKAGES AND SETUP ENVIRONMENT
import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils

# Training Parameters
batch_size = 128
data_path = '/tmp/data/mnist'
num_classes = 10
# MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# 1.2 DOWNLOAD DATASET
# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

# Try to download MNIST dataset
try:
    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
except Exception as e:
    print(f"Error downloading MNIST dataset: {e}")
    print("Attempting to download using alternative method...")
    # Uncomment the following lines if the direct download fails
    # !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    # !tar -zxvf MNIST.tar.gz
    # mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transform)

# Reduce the dataset size
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

# 1.3 CREATE DATALOADERS
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# 2.3 LATENCY CODING OF MNIST
def convert_to_time(data, tau=5, threshold=0.01):
    spike_time = tau * torch.log(data / (data - threshold))
    return spike_time












