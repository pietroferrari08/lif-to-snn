
import csv
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mnist_snn_model import MNISTSNN


SEED = 0
NUM_STEPS = 30
NUM_EPOCHS = 3
BATCH_SIZE = 32

CHECKPOINT = Path("checkpoints/mnist_val_reference.pth")

# =====================================================
# 1. DATASET: STESSA DIVISIONE DEL FILE 10
# =====================================================

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

mask = (mnist_train.targets == 0) | (mnist_train.targets == 1)
indices = torch.where(mask)[0]

generator = torch.Generator().manual_seed(SEED)
permutation = torch.randperm(len(indices), generator=generator)
shuffled_indices = indices[permutation]

val_size = int(0.10 * len(indices))

val_indices = shuffled_indices[:val_size]
train_indices = shuffled_indices[val_size:]

train_data = Subset(mnist_train, train_indices.tolist())
val_data = Subset(mnist_train, val_indices.tolist())

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# 2. RETE DI RIFERIMENTO
# =====================================================

torch.manual_seed(SEED)

reference_model = MNISTSNN(
    beta=0.8,
    threshold=1.0,
    num_steps=NUM_STEPS,
    hidden_size=32
)

# Carichiamo i pesi, se già disponibili
if CHECKPOINT.exists():

    saved = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    reference_model.fc1.load_state_dict(saved["fc1"])
    reference_model.fc2.load_state_dict(saved["fc2"])

    print("Riferimento caricato dal checkpoint.")

else:

    print("Addestramento della rete di riferimento.")

    optimizer = torch.optim.Adam(
        reference_model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    reference_model.train()

    for epoch in range(NUM_EPOCHS):

        total_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            spike_counts = reference_model(images)
            loss = criterion(spike_counts, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        print(
            f"Epoca {epoch + 1}: "
            f"Loss={total_loss / len(train_data):.4f}"
        )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "fc1": reference_model.fc1.state_dict(),
            "fc2": reference_model.fc2.state_dict()
        },
        CHECKPOINT
    )

    print("Riferimento salvato.")


# =====================================================
# 3. VALUTAZIONE A PESI FISSI
# =====================================================

def evaluate_fixed(beta, threshold):

    model = MNISTSNN(
        beta=beta,
        threshold=threshold,
        num_steps=NUM_STEPS,
        hidden_size=32
    )

    # Copiamo SOLO i pesi sinaptici:
    # non copiamo beta e threshold della rete originale
    model.fc1.load_state_dict(
        reference_model.fc1.state_dict()
    )

    model.fc2.load_state_dict(
        reference_model.fc2.state_dict()
    )

    model.eval()

    # Stessa realizzazione casuale degli spike
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

    return {
        "accuracy": correct / total,
        "ties": ties / total,
        "output_rate": total_spikes / (total * 2 * NUM_STEPS)
    }


# =====================================================
# 4. CONFRONTO
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

    metrics = evaluate_fixed(beta, threshold)

    results.append({
        "name": name,
        "beta": beta,
        "threshold": threshold,
        **metrics
    })

    print(
        f"{name}: "
        f"Accuracy={metrics['accuracy']:.2%}, "
        f"Ties={metrics['ties']:.2%}, "
        f"Output rate={metrics['output_rate']:.2%}"
    )


# =====================================================
# 5. SALVATAGGIO RISULTATI E GRAFICO
# =====================================================

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

with open(
    "results/fixed_weights.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "name", "beta", "threshold",
            "accuracy", "ties", "output_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

names = [r["name"] for r in results]
rates = [r["output_rate"] * 100 for r in results]

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(names, rates)
ax.set_ylabel("Output firing rate (%)")
ax.set_title("Effetto dei parametri LIF a pesi fissi")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig("figures/fixed_weights_spiking.png", dpi=200)

plt.show()
