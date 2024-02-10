import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt



num_steps = 100
U = 0.9
U_trace = []

def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
	tau = R*C
	U = U + (time_step/tau)*(-U + I*R)
	return U

for step in range(num_steps):
	U_trace.append(U)
	U = leaky_integrate_neuron(U)

plt.plot(U_trace)
plt.title("Leaky Neuron Model")
plt.xlabel("Time Stemp")
plt.ylabel("Membrane Potential")
plt.show()
