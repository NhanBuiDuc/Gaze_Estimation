import torch
import snntorch as snn
from snntorch import surrogate

class Net(torch.nn.Module):
    """Simple spiking neural network using only Leaky neurons."""
    
    def __init__(self, timesteps, hidden, output_size):
        super().__init__()
        
        self.timesteps = timesteps
        self.hidden = hidden
        self.output_size = output_size
        spike_grad = surrogate.fast_sigmoid()
        
        # Define Leaky modules for input and hidden layers
        self.lif_in = snn.Leaky(beta=torch.rand(self.hidden), threshold=torch.rand(self.hidden), learn_beta=True, spike_grad=spike_grad)
        self.lif_hidden = snn.Leaky(beta=torch.rand(self.hidden), threshold=torch.rand(self.hidden), learn_beta=True, spike_grad=spike_grad)
        
        # Linear layer to transition from hidden size to output size
        self.linear = torch.nn.Linear(self.hidden, self.output_size)
        
        # Output layer with Leaky neuron dynamics producing output_size units
        self.lif_out = snn.Leaky(beta=torch.rand(self.output_size), threshold=torch.ones(self.output_size), learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")

    def forward(self, x):
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_out = self.lif_out.init_leaky()

        mem_out_rec = []

        for step in range(self.timesteps):
            x_timestep = x[step, :, :]

            # Apply input layer Leaky neuron dynamics
            spk_in, mem_1 = self.lif_in(x_timestep, mem_1)
            
            # Apply hidden layer Leaky neuron dynamics
            spk_hidden, mem_2 = self.lif_hidden(spk_in, mem_2)
            
            # Apply linear transformation to transition dimensions
            lin_out = self.linear(spk_hidden)
            
            # Apply output layer Leaky neuron dynamics
            spk_out, mem_out = self.lif_out(lin_out, mem_out)

            mem_out_rec.append(mem_out)

        return torch.stack(mem_out_rec)