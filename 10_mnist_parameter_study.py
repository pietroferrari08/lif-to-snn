import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mnist_snn_model import MNISTSNN

import matplotlib.pyplot as plt
from pathlib import Path

SEED = 0
BATCH_SIZE = 32
NUM_STEPS = 30
NUM_EPOCHS = 3

# 1. Carichiamo MNIST training
mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 2. Selezioniamo tutti gli 0 e gli 1
mask = (mnist_train.targets == 0) | (mnist_train.targets == 1)

indices = torch.where(mask)[0]

# 3. Mescoliamo gli indici in modo riproducibile
generator = torch.Generator().manual_seed(SEED)

permutation = torch.randperm(
    len(indices),
    generator=generator
)

shuffled_indices = indices[permutation]

# 4. Separiamo training e validation
val_size = int(0.10 * len(indices))

val_indices = shuffled_indices[:val_size]
train_indices = shuffled_indices[val_size:]

train_data = Subset(
    mnist_train,
    train_indices.tolist()
)

val_data = Subset(
    mnist_train,
    val_indices.tolist()
)

# 5. Creiamo i DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# ESPERIMENTO SU UNA CONFIGURAZIONE
# =====================================================

def run_experiment(beta, threshold):

    # Stessa inizializzazione per tutti gli esperimenti
    torch.manual_seed(SEED)

    model = MNISTSNN(
        beta=beta,
        threshold=threshold,
        num_steps=NUM_STEPS,
        hidden_size=32
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    # Stesso ordine dei batch per ogni configurazione
    experiment_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )

    # TRAINING
    model.train()

    for epoch in range(NUM_EPOCHS):

        total_loss = 0

        for images, labels in experiment_loader:

            optimizer.zero_grad()

            spike_counts = model(images)

            loss = criterion(spike_counts, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_data)

        print(f"  Epoca {epoch+1}: Loss={avg_loss:.4f}")

    # VALIDATION
    model.eval()

    # Stesso campionamento casuale della validation
    torch.manual_seed(SEED + 1000)

    correct = 0
    ties = 0
    total = 0
    total_spikes = 0

    with torch.no_grad():

        for images, labels in val_loader:

            spike_counts = model(images)

            predictions = spike_counts.argmax(dim=1)

            is_tie = spike_counts[:, 0] == spike_counts[:, 1]

            correct += (
                (predictions == labels) & ~is_tie
            ).sum().item()

            ties += is_tie.sum().item()
            total += labels.size(0)

            total_spikes += spike_counts.sum().item()

    accuracy = correct / total
    tie_rate = ties / total

    # Spike medi per neurone di uscita e time step
    output_rate = total_spikes / (total * 2 * NUM_STEPS)

    return accuracy, tie_rate, output_rate



# =====================================================
# CONFRONTO TRA CONFIGURAZIONI
# =====================================================

configurations = [
    ("Riferimento", 0.8, 1.0),
    ("Beta 0.6", 0.6, 1.0),
    ("Beta 0.9", 0.9, 1.0),
    ("Threshold 0.8", 0.8, 0.8),
    ("Threshold 1.2", 0.8, 1.2)
]

results = []

for name, beta, threshold in configurations:

    print(f"\n=== {name} ===")

    accuracy, tie_rate, output_rate = run_experiment(
        beta,
        threshold
    )

    results.append({
        "name": name,
        "beta": beta,
        "threshold": threshold,
        "accuracy": accuracy,
        "tie_rate": tie_rate,
        "output_rate": output_rate
    })

print("\n=== RISULTATI FINALI ===")

for result in results:

    print(
        f"{result['name']}: "
        f"Accuracy={result['accuracy']:.2%}, "
        f"Ties={result['tie_rate']:.2%}, "
        f"Output rate={result['output_rate']:.2%}"
    )

# =====================================================
# GRAFICI DEI RISULTATI
# =====================================================

figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

names = [r["name"] for r in results]

failures = [
    round((1 - r["accuracy"]) * len(val_data))
    for r in results
]

rates = [
    r["output_rate"] * 100
    for r in results
]

# GRAFICO 1: ERRORI

fig1, ax1 = plt.subplots(figsize=(10, 4))

ax1.bar(names, failures)
ax1.set_ylabel("Immagini non corrette (inclusi pareggi)")
ax1.set_title("Confronto delle configurazioni SNN")
ax1.tick_params(axis="x", rotation=20)

fig1.tight_layout()
fig1.savefig(figures_dir / "parameter_accuracy.png", dpi=200)


# GRAFICO 2: FIRING RATE

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.bar(names, rates)
ax2.set_ylabel("Output firing rate (%)")
ax2.set_title("Attività neuronale dopo il training")
ax2.set_ylim(0, 45)
ax2.tick_params(axis="x", rotation=20)

fig2.tight_layout()
fig2.savefig(figures_dir / "parameter_spiking.png", dpi=200)


# VERIFICA FILE SALVATI

print("\nGrafici salvati:")

print((figures_dir / "parameter_accuracy.png").resolve())
print((figures_dir / "parameter_spiking.png").resolve())

# Mostriamo entrambi i grafici alla fine
plt.show()