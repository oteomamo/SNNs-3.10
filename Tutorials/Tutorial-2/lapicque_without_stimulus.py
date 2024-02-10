import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

time_step = 1e-3
R = 5
C = 1e-3

# leaky integrate and fire neuron, tau=5e-3
lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

# Initialize membrane potential, input, and output
mem = torch.ones(1) * 0.9  # U=0.9 at t=0
num_steps = 200
cur_in = torch.zeros(num_steps)  # I=0 for all t
spk_out = torch.zeros(1)  # initialize output spikes

# A list to store a recording of membrane potential
mem_rec = [mem]

# pass updated value of mem and cur_in[step]=0 at every time step
for step in range(num_steps):
  spk_out, mem = lif1(cur_in[step], mem)

  # Store recordings of membrane potential
  mem_rec.append(mem)

# convert the list of tensors into one tensor
mem_rec = torch.stack(mem_rec)

# pre-defined plotting function
plt.plot(mem_rec)
plt.title("Lapicque's Neuron Model Without Stimulus")
plt.xlabel("Time step")
plt.ylabel("Membrane Potential")
plt.show()



# Initialize input current pulse
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.1), 0)  # input current turns on at t=10

# Initialize membrane, output and recordings
mem = torch.zeros(1)  # membrane potential of 0 at t=0
spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
mem_rec = [mem]

# pass updated value of mem and cur_in[step] at every time step
for step in range(num_steps):
  spk_out, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)

# crunch -list- of tensors into one tensor
mem_rec = torch.stack(mem_rec)

def plot_step_current_response(current_input, membrane_record, time_step):
    plt.figure(figsize=(12, 5))

    # Plot current input
    plt.subplot(1, 2, 1)
    plt.plot(current_input.numpy())
    plt.title("Input Current")
    plt.xlabel("Time step")
    plt.ylabel("Input Current (I_in)")

    # Plot membrane potential
    plt.subplot(1, 2, 2)
    plt.plot(membrane_record.detach().numpy())
    plt.title("Membrane Potential")
    plt.xlabel("Time step")
    plt.ylabel("Membrane Potential (Umem)")

    plt.suptitle("Lapicque's Neuron Model With Step Input", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


plot_step_current_response(cur_in, mem_rec, 10)
