
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen



class MNISTSNN(nn.Module):

    def __init__(
        self,
        beta=0.8,
        threshold=1.0,
        num_steps=30,
        hidden_size=32
    ):

        super().__init__()

        self.num_steps = num_steps
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(784, hidden_size, bias=False)

        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism="zero",
            reset_delay=False
        )

        self.fc2 = nn.Linear(hidden_size, 2, bias=False)

        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism="zero",
            reset_delay=False
        )

    def forward(self, images):

        spike_train = spikegen.rate(
            images,
            num_steps=self.num_steps
        )

        spikes_flat = spike_train.flatten(start_dim=2)

        batch_size = images.shape[0]

        mem1 = images.new_zeros(batch_size, self.hidden_size)
        mem2 = images.new_zeros(batch_size, 2)

        output_spikes = []

        for step in range(self.num_steps):

            spikes_t = spikes_flat[step]

            current_hidden = self.fc1(spikes_t)
            spk_hidden, mem1 = self.lif1(current_hidden, mem1)

            current_output = self.fc2(spk_hidden)
            spk_output, mem2 = self.lif2(current_output, mem2)

            output_spikes.append(spk_output)

        spike_history = torch.stack(output_spikes, dim=0)

        return spike_history.sum(dim=0)