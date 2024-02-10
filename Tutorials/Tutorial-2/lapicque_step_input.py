import torch
import matplotlib.pyplot as plt
from lapicque_without_stimulus import lif1
num_steps = 200

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

