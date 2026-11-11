
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# 1. PARAMETRI
# =====================================================

SEED = 0

NUM_STEPS = 30
HIDDEN_SIZE = 128
NUM_CLASSES = 10

BETA = 0.8
THRESHOLD = 1.0

BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_MODEL = True

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CHECKPOINT = Path("checkpoints/mnist_snn_10_classes.pth")

torch.manual_seed(SEED)

print("Dispositivo:", DEVICE)


# =====================================================
# 2. DATASET MNIST COMPLETO
# =====================================================

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED)
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Training images:", len(train_data))
print("Test images:", len(test_data))


# =====================================================
# 3. MODELLO SNN
# =====================================================

class MNIST10SNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            784,
            HIDDEN_SIZE,
            bias=False
        )

        self.lif1 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            reset_mechanism="zero",
            reset_delay=False
        )

        self.fc2 = nn.Linear(
            HIDDEN_SIZE,
            NUM_CLASSES,
            bias=False
        )

        self.lif2 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
            reset_mechanism="zero",
            reset_delay=False
        )

    def forward(self, images):

        # [B, 1, 28, 28] -> [T, B, 1, 28, 28]
        spike_train = spikegen.rate(
            images,
            num_steps=NUM_STEPS
        )

        # [T, B, 784]
        spikes_flat = spike_train.flatten(start_dim=2)

        batch_size = images.shape[0]

        mem1 = images.new_zeros(
            batch_size,
            HIDDEN_SIZE
        )

        mem2 = images.new_zeros(
            batch_size,
            NUM_CLASSES
        )

        output_spikes = []

        for t in range(NUM_STEPS):

            current_hidden = self.fc1(spikes_flat[t])

            spk_hidden, mem1 = self.lif1(
                current_hidden,
                mem1
            )

            current_output = self.fc2(spk_hidden)

            spk_output, mem2 = self.lif2(
                current_output,
                mem2
            )

            output_spikes.append(spk_output)

        # [T, B, 10] -> [B, 10]
        spike_counts = torch.stack(
            output_spikes,
            dim=0
        ).sum(dim=0)

        return spike_counts


model = MNIST10SNN().to(DEVICE)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Parametri allenabili:", num_parameters)


# =====================================================
# 4. TRAINING
# =====================================================

if TRAIN_MODEL:

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(NUM_EPOCHS):

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            spike_counts = model(images)

            loss = criterion(
                spike_counts,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = spike_counts.argmax(dim=1)

            # Pareggio tra i due conteggi massimi
            top_two = spike_counts.topk(
                2,
                dim=1
            ).values

            is_tie = top_two[:, 0] == top_two[:, 1]

            correct += (
                (predictions == labels) & ~is_tie
            ).sum().item()

            total += labels.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(
                    f"Epoca {epoch+1}, "
                    f"batch {batch_idx+1}/{len(train_loader)}"
                )

        avg_loss = total_loss / total
        accuracy = correct / total

        print(
            f"Epoca {epoch+1}: "
            f"Loss={avg_loss:.4f}, "
            f"Train Accuracy={accuracy:.2%}"
        )

    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        CHECKPOINT
    )

    print("Modello salvato:", CHECKPOINT)

else:

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint non trovato: {CHECKPOINT}"
        )

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE,
            weights_only=True
        )
    )

    print("Checkpoint caricato.")


# =====================================================
# 5. TEST
# =====================================================

model.eval()

torch.manual_seed(SEED + 1000)

correct = 0
ties = 0
total = 0
total_spikes = 0

# 10 classi previste + una colonna per i pareggi
confusion = torch.zeros(
    10,
    11,
    dtype=torch.long
)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        spike_counts = model(images)

        predictions = spike_counts.argmax(dim=1)

        top_two = spike_counts.topk(
            2,
            dim=1
        ).values

        is_tie = top_two[:, 0] == top_two[:, 1]

        correct += (
            (predictions == labels) & ~is_tie
        ).sum().item()

        ties += is_tie.sum().item()
        total += labels.size(0)

        total_spikes += spike_counts.sum().item()

        # Per la confusion matrix:
        # colonna 10 = pareggio
        for true_label, prediction, tie in zip(
            labels.cpu().tolist(),
            predictions.cpu().tolist(),
            is_tie.cpu().tolist()
        ):

            predicted_column = (
                10 if tie else prediction
            )

            confusion[
                true_label,
                predicted_column
            ] += 1


# =====================================================
# 6. RISULTATI
# =====================================================

accuracy = correct / total
tie_rate = ties / total

output_rate = total_spikes / (
    total * NUM_CLASSES * NUM_STEPS
)

errors = total - correct - ties

print("\n=== RISULTATI TEST MNIST 10 CLASSI ===")

print(f"Accuracy: {accuracy:.2%}")
print(f"Pareggi: {ties} ({tie_rate:.2%})")
print(f"Errori senza pareggio: {errors}")
print(f"Output firing rate: {output_rate:.2%}")

print("\nConfusion matrix:")
print(confusion)


# =====================================================
# 7. GRAFICO MATRICE DI CONFUSIONE
# =====================================================

Path("figures").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 8))

matrix = confusion.numpy()

im = ax.imshow(
    matrix,
    cmap="Blues"
)

ax.set_xticks(range(11))
ax.set_xticklabels(
    list(range(10)) + ["Tie"]
)

ax.set_yticks(range(10))
ax.set_yticklabels(range(10))

ax.set_xlabel("Classe predetta")
ax.set_ylabel("Classe reale")
ax.set_title("SNN — MNIST 10 classi")

fig.colorbar(
    im,
    ax=ax,
    label="Numero immagini"
)

fig.tight_layout()

fig.savefig(
    "figures/mnist10_confusion.png",
    dpi=200
)

plt.show()
