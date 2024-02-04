import torch
from snntorch import spikegen

def create_rate_coded_vector(num_steps, p_spike=0.5):
    """Create and rate code a vector with a specified spike probability."""
    # create vector filled with p_spike
    raw_vector = torch.ones(num_steps) * p_spike

    # pass each sample through a Bernoulli trial
    rate_coded_vector = torch.bernoulli(raw_vector)
    print(f"Converted vector: {rate_coded_vector}")
    print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")

def rate_code_mnist_data(train_loader, num_steps):
    """Generate rate-coded samples for MNIST data."""
    data_it, targets_it = next(iter(train_loader))

    # Spiking Data
    spike_data = spikegen.rate(data_it, num_steps=num_steps)
    # print(spike_data.size())
    return spike_data, targets_it
