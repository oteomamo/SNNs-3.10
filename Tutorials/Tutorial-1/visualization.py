import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import display, HTML
import torch  # Assuming spike_data and targets_it are passed as arguments
from mnist_setup import convert_to_time

def visualize_spike_data(spike_data, targets_it, num_steps=100, ffmpeg_path=None, gain=0):
    """Visualize and animate spike data."""
    # Index into a single sample from the batch dimension
    spike_data_sample = spike_data[:, 0, 0]
    print(f"Spike data sample size: {spike_data_sample.size()}")

    # Animation
    fig, ax = plt.subplots()
    anim = splt.animator(spike_data_sample, fig, ax)
    if ffmpeg_path:  # Uncomment if ffmpeg path needs to be set
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # Display the animation
    display(HTML(anim.to_html5_video()))

    # Optional: Save the animation
    anim.save(f"spike_mnist_test.mp4")

    # Print the corresponding target label
    print(f"The corresponding target is: {targets_it[0]}")

    # Average spikes over time and reconstruct input images
    plt.figure(facecolor="w")
    plt.subplot(1,2,1)
    plt.imshow(spike_data_sample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    plt.axis('off')
    plt.title(f'Average Spike Visualization')
    plt.title('Gain = 1')

    # plt.subplot(1,2,2)
    # plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
    # plt.axis('off')
    # plt.title(f'Spike Visualization With Grain = 0.25')
    # plt.title('Gain = 0.25')

    plt.show()

    # Reshape spike_data_sample for raster plot
    # spike_data_sample2 = spike_data_sample.reshape((num_steps, -1))

    # Generate raster plot
    # fig = plt.figure(facecolor="w", figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # splt.raster(spike_data_sample2, ax, s=1.5, c="black")

    # plt.title("Input Layer")
    # plt.xlabel("Time step")
    # plt.ylabel("Neuron Number")
    # plt.show()
    # idx = 210  # index into the 210th neuron

    # fig = plt.figure(facecolor="w", figsize=(8, 1))
    # ax = fig.add_subplot(111)

    # splt.raster(spike_data_sample2.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")

    # plt.title("Input Neuron")
    # plt.xlabel("Time step")
    # plt.yticks([])
    # plt.show()

def visualize_latency_coding():
    raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
    spike_times = convert_to_time(raw_input)  # Assuming convert_to_time is accessible

    plt.plot(raw_input, spike_times)
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.title('Latency Coding Visualization')
    plt.show()

#  visualize_latency_coding()
