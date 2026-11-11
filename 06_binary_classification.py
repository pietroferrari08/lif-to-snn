import torch
import torch.nn as nn
import snntorch as snn

spikes_1 = []
spikes_2 = []
output_spikes = []

#crea un oggetto che rappresenta un neurone LIF   
lif = snn.Leaky( 
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)

patterns = [
    (torch.tensor([0.60, 0.25]), 0),
    (torch.tensor([0.25, 0.60]), 1)
]

# Due pesi, uno per ciascun neurone
weights = torch.tensor([0.60, 0.25])

# Stesso ingresso per entrambi
x = torch.ones(2)


connection = nn.Linear(2, 2, bias=False)

lif_out = snn.Leaky(
    beta=0.8,
    threshold=1,
    reset_mechanism="zero",
    reset_delay=False
)

# Imposta il seed per la generazione casuale dei numeri
torch.manual_seed(0)

# Inizializza i pesi della connessione in modo casuale tra 0.2 e 0.8
with torch.no_grad():
    connection.weight.uniform_(0.2, 0.8) #il _ indica che la funzione modifica direttamente i pesi senza restituire un nuovo tensore



def simulate(currents):
    # Stato iniziale dei due neuroni d'ingresso
    mem = torch.zeros(2)

    # Stato iniziale dei due neuroni d'uscita
    mem_out = torch.zeros(2)

    output_spikes = []

    for step in range(30):

        spk_in, mem = lif(currents, mem)

        current_out = connection(spk_in)

        spk_out, mem_out = lif_out(current_out, mem_out)

        output_spikes.append(spk_out)

    spike_history = torch.stack(output_spikes, dim=0) #Con dim=0 diciamo a PyTorch di creare la nuova dimensione prima di quelle già esistenti

    spike_counts = spike_history.sum(dim=0) #Nel nostro caso dim=0 rappresenta il tempo: sommiamo verticalmente tutti i 30 passi, mantenendo separate le due colonne dei neuroni.

    return spike_counts

loss_fn = nn.CrossEntropyLoss() #crea un oggetto che misura quanto i punteggi assegnati alle classi favoriscano quella corretta.

optimizer = torch.optim.SGD(connection.parameters(), lr=0.05) #crea un oggetto che aggiorna i pesi della connessione in base al gradiente calcolato dalla loss function

for epoch in range(100):

    optimizer.zero_grad()

    total_loss = 0.0

    for currents, true_class in patterns:

        counts = simulate(currents)

        target = torch.tensor(true_class, dtype=torch.long ) #

        loss = loss_fn(counts, target)

        total_loss = total_loss + loss

    total_loss = total_loss / len(patterns)

    total_loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoca {epoch+1} | "
            f"Loss: {total_loss.item():.4f}"
        )

print("Pesi finali:" , connection.weight.detach())

#verifichiamo le prestazioni del modello sui pattern di input
#ovvero, non usiamo più semplicemente i pattern di addestramento, ma anche altri pattern simili a quelli di addestramento, per vedere se il modello è in grado di generalizzare.

test_patterns = [
    (torch.tensor([0.55, 0.30]), 0),
    (torch.tensor([0.50, 0.35]), 0),
    (torch.tensor([0.35, 0.50]), 1),
    (torch.tensor([0.30, 0.55]), 1)
]

with torch.no_grad():
    correct = 0

    for currents, true_class in test_patterns:
        counts = simulate(currents) 

        if counts[0] == counts[1]:
            predicted_class = None
        else:
            predicted_class = counts.argmax().item()

        if predicted_class == true_class:
            correct += 1

        print(
            f"Input: {currents.tolist()} | "
            f"Spike: {counts.tolist()} | "
            f"Reale: {true_class} | "
            f"Prevista: {predicted_class}"
        )

    print(f"Test accuracy: {correct}/{len(test_patterns)}")
    

#testiamo il modello con un nuovo set di pattern di input, che rappresentano due schemi di stress diversi. Il primo schema ha un livello di stress più alto nel primo neurone, mentre il secondo schema ha un livello di stress più alto nel secondo neurone. L'obiettivo è verificare se il modello è in grado di distinguere tra i due schemi di stress e classificare correttamente l'input.


    stress_patterns = [
    (torch.tensor([0.50, 0.49]), 0),
    (torch.tensor([0.49, 0.50]), 1),
    (torch.tensor([0.20, 0.10]), 0),
    (torch.tensor([0.10, 0.20]), 1)
]
    
with torch.no_grad():
    correct = 0

    for currents, true_class in stress_patterns:
        counts = simulate(currents) 

        if counts[0] == counts[1]:
            predicted_class = None
        else:
            predicted_class = counts.argmax().item()

        if predicted_class == true_class:
            correct += 1

        print(
            f"Input: {currents.tolist()} | "
            f"Spike: {counts.tolist()} | "
            f"Reale: {true_class} | "
            f"Prevista: {predicted_class}"
        )

    print(f"Test accuracy: {correct}/{len(test_patterns)}")