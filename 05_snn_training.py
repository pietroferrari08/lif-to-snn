import torch
import torch.nn as nn
import snntorch as snn

#crea un oggetto che rappresenta un neurone LIF   
lif = snn.Leaky( 
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)

# Pesi fissi verso i primi due neuroni
weights = torch.tensor([0.25, 0.60])

# Stesso ingresso per entrambi
x = torch.ones(2)

connection = nn.Linear(2, 1, bias=False)

lif_out = snn.Leaky(
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)

# Pesi apprendibili verso il neurone di uscita
connection = nn.Linear(2, 1, bias=False)

with torch.no_grad():
    connection.weight.copy_( torch.tensor([[0.6, 0.6]]))

# Ottimizzatore
optimizer = torch.optim.SGD(
    connection.parameters(),
    lr=0.5
)

target_rate = torch.tensor(0.30)

# Funzione di errore
loss_fn = nn.MSELoss()


for epoch in range(50):
    spike_counts_in = torch.zeros(2)

    # Reset dello stato per una nuova simulazione
    mem = torch.zeros(2)
    mem_out = torch.zeros(1)

    spikes_out_tensors = []
    optimizer.zero_grad()

    # Simulazione della SNN
    for step in range(30):

        current = x * weights
        spk_in, mem = lif(current, mem)

        current_out = connection(spk_in)
        spk_out, mem_out = lif_out(current_out, mem_out)

        spikes_out_tensors.append(spk_out)

        spike_counts_in += spk_in.detach()

    # Predizione: firing rate sui 30 passi
    prediction = torch.stack(spikes_out_tensors).mean()

    # Errore rispetto al firing rate desiderato
    loss = loss_fn(prediction,target_rate)

    loss.backward()

    optimizer.step()

    if epoch == 0:
        print("Pesi ingresso:", weights)
        print("Spike neuroni iniziali:", spike_counts_in.tolist())
        print("Pesi connessione:", connection.weight.detach())

    if (epoch + 1) % 5 == 0:
        print("Gradienti:", connection.weight.grad)
        print("Epoca:", epoch + 1)
        print("Firing rate:", prediction.item())
        print("Loss:", loss.item())
        print("Pesi:", connection.weight.detach())
