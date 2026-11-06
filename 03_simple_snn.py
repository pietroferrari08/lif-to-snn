import torch
import torch.nn as nn
import snntorch as snn

spikes_1 = []
spikes_2 = []
spikes_out = []

#crea un oggetto che rappresenta un neurone LIF   
lif = snn.Leaky( 
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)


# Due pesi, uno per ciascun neurone
weights = torch.tensor([0.25, 0.60])

# Stesso ingresso per entrambi
x = torch.ones(2)

# Stato iniziale dei due neuroni
mem = torch.zeros(2)

connection = nn.Linear(2, 1, bias=False)

lif_out = snn.Leaky(
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)

with torch.no_grad():
    connection.weight.copy_(torch.tensor([[0.6, 1.01]]))

mem_out = torch.zeros(1)

for step in range(30):
    current = x * weights
    spk_in, mem = lif(current, mem) #compaiono due variabili perché la funzione restituisce due risultati
    '''Python permette di definire un metodo chiamato __call__, che viene eseguito quando usiamo le parentesi tonde direttamente sull'oggetto'''
    current_out = connection(spk_in)
    spk_out, mem_out = lif_out(current_out,mem_out)
    spikes_1.append(spk_in[0].item())
    spikes_2.append(spk_in[1].item())
    spikes_out.append(spk_out.item())
 #item permette di estrarre l'effettivo valore da spk e mem che sono tensori

print("Neurone 1:", spikes_1)
print("Neurone 2:", spikes_2)
print("Neurone out: ",spikes_out)

print("Spike neurone 1:", sum(spikes_1))
print("Spike neurone 2:", sum(spikes_2))
print("Spike neurone out:", sum(spikes_out))


