import torch
from lapicque_without_stimulus import *

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
plt.title( "Lapicque's Neuron Model Without Stimulus")
plt.xlabel("Time step")
plt.ylabel("Membrane Potential")
plt.show()
