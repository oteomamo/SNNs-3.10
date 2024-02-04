import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import display, HTML
import torch  # Assuming spike_data and targets_it are passed as arguments

def visualize_spike_data(spike_data, targets_it, num_steps=100, ffmpeg_path=None):
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

    plt.show()

