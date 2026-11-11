
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================
# 1. CONFIGURAZIONE
# =====================================================

SEED = 0
NUM_STEPS = 30
NUM_EPOCHS = 3
BATCH_SIZE = 32

HIDDEN_SIZE = 32
BETA = 0.8
THRESHOLD = 1.0
LEARNING_RATE = 0.001

# True = addestra da zero e salva i pesi
# False = carica i pesi salvati ed esegue solo il test
TRAIN_MODEL = False

NUM_TEST_RUNS = 5

CHECKPOINT_PATH = Path("checkpoints/mnist_snn_01.pth")

device = torch.device("cpu")
torch.manual_seed(SEED)


# =====================================================
# 2. DATASET MNIST
# =====================================================

mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

mnist_test = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

# Selezioniamo TUTTI gli 0 e gli 1
train_mask = (
    (mnist_train.targets == 0) |
    (mnist_train.targets == 1)
)

test_mask = (
    (mnist_test.targets == 0) |
    (mnist_test.targets == 1)
)

train_indices = torch.where(train_mask)[0]
test_indices = torch.where(test_mask)[0]

train_data = Subset(
    mnist_train,
    train_indices.tolist()
)

test_data = Subset(
    mnist_test,
    test_indices.tolist()
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Immagini training:", len(train_data))
print("Immagini test:", len(test_data))

print(
    "Classi training:",
    torch.bincount(mnist_train.targets[train_indices], minlength=2)
)

print(
    "Classi test:",
    torch.bincount(mnist_test.targets[test_indices], minlength=2)
)


# =====================================================
# 3. ARCHITETTURA SNN
# =====================================================

# 784 pixel -> 32 neuroni nascosti
fc1 = nn.Linear(
    784,
    HIDDEN_SIZE,
    bias=False
).to(device)

lif1 = snn.Leaky(
    beta=BETA,
    threshold=THRESHOLD,
    reset_mechanism="zero",
    reset_delay=False
).to(device)

# 32 neuroni nascosti -> 2 neuroni di uscita
fc2 = nn.Linear(
    HIDDEN_SIZE,
    2,
    bias=False
).to(device)

lif2 = snn.Leaky(
    beta=BETA,
    threshold=THRESHOLD,
    reset_mechanism="zero",
    reset_delay=False
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(fc1.parameters()) + list(fc2.parameters()),
    lr=LEARNING_RATE
)


# =====================================================
# 4. FORWARD DELLA SNN
# =====================================================

def forward_batch(images):

    spike_train = spikegen.rate(
        images,
        num_steps=NUM_STEPS
    )

    # [T, B, 1, 28, 28] -> [T, B, 784]
    spikes_flat = spike_train.flatten(start_dim=2)

    batch_size = images.shape[0]

    # Memorie indipendenti per ogni immagine
    mem1 = images.new_zeros(batch_size, HIDDEN_SIZE)
    mem2 = images.new_zeros(batch_size, 2)

    output_spikes = []

    for step in range(NUM_STEPS):

        spikes_t = spikes_flat[step]

        current_hidden = fc1(spikes_t)
        spk_hidden, mem1 = lif1(current_hidden, mem1)

        current_output = fc2(spk_hidden)
        spk_output, mem2 = lif2(current_output, mem2)

        output_spikes.append(spk_output)

    spike_history = torch.stack(
        output_spikes,
        dim=0
    )

    # Sommiamo SOLO lungo il tempo
    spike_counts = spike_history.sum(dim=0)

    return spike_counts


# =====================================================
# 5. TRAINING O CARICAMENTO
# =====================================================

if TRAIN_MODEL:

    fc1.train()
    fc2.train()

    print("\n=== INIZIO TRAINING ===")

    for epoch in range(NUM_EPOCHS):

        total_loss = 0.0
        correct = 0
        ties = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            spike_counts = forward_batch(images)

            # Loss
            loss = criterion(spike_counts, labels)

            # BPTT e aggiornamento pesi
            loss.backward()
            optimizer.step()

            # Statistiche
            predictions = spike_counts.argmax(dim=1)

            is_tie = (
                spike_counts[:, 0] ==
                spike_counts[:, 1]
            )

            correct += (
                (predictions == labels) & ~is_tie
            ).sum().item()

            ties += is_tie.sum().item()
            total += labels.size(0)

            total_loss += loss.item() * labels.size(0)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
            f"Loss={total_loss / total:.4f}, "
            f"Accuracy={correct / total:.2%}, "
            f"Ties={ties / total:.2%}"
        )

    # Salviamo i pesi e la configurazione
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "fc1": fc1.state_dict(),
            "fc2": fc2.state_dict(),
            "num_steps": NUM_STEPS,
            "hidden_size": HIDDEN_SIZE,
            "beta": BETA,
            "threshold": THRESHOLD
        },
        CHECKPOINT_PATH
    )

    print("\nPesi salvati in:", CHECKPOINT_PATH)

else:

    print("\n=== CARICAMENTO MODELLO ===")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    # Verifica che l'architettura sia compatibile
    if (
        checkpoint["num_steps"] != NUM_STEPS
        or checkpoint["hidden_size"] != HIDDEN_SIZE
        or checkpoint["beta"] != BETA
        or checkpoint["threshold"] != THRESHOLD
    ):
        raise ValueError(
            "Configurazione diversa da quella del modello salvato."
        )

    fc1.load_state_dict(checkpoint["fc1"])
    fc2.load_state_dict(checkpoint["fc2"])

    print("Pesi caricati da:", CHECKPOINT_PATH)


# =====================================================
# 6. FUNZIONE DI VALUTAZIONE
# =====================================================

def evaluate(loader):

    fc1.eval()
    fc2.eval()

    total_loss = 0.0

    # Righe: classe reale 0, 1
    # Colonne: previsione 0, 1, pareggio
    confusion = torch.zeros(2, 3, dtype=torch.long)

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            spike_counts = forward_batch(images)

            loss = criterion(spike_counts, labels)

            predictions = spike_counts.argmax(dim=1)

            is_tie = (
                spike_counts[:, 0] ==
                spike_counts[:, 1]
            )

            # Assegniamo ai pareggi il codice 2
            predictions = predictions.clone()
            predictions[is_tie] = 2

            # Conteggio delle combinazioni classe/previsione
            encoded = labels * 3 + predictions

            batch_confusion = torch.bincount(
                encoded.cpu(),
                minlength=6
            ).reshape(2, 3)

            confusion += batch_confusion

            total_loss += loss.item() * labels.size(0)

    total = confusion.sum().item()

    correct = (
        confusion[0, 0].item() +
        confusion[1, 1].item()
    )

    ties = confusion[:, 2].sum().item()

    errors = (
        confusion[0, 1].item() +
        confusion[1, 0].item()
    )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "ties": ties / total,
        "errors": errors,
        "total": total,
        "confusion": confusion
    }


# =====================================================
# 7. TEST FINALE SU TUTTE LE IMMAGINI 0 E 1
# =====================================================

print("\n=== TEST FINALE ===")

for run in range(NUM_TEST_RUNS):

    # Nuovo campionamento degli spike a ogni test
    torch.manual_seed(SEED + run)

    results = evaluate(test_loader)

    print(
        f"Test {run + 1}: "
        f"Loss={results['loss']:.4f}, "
        f"Accuracy={results['accuracy']:.2%}, "
        f"Ties={results['ties']:.2%}, "
        f"Errors={results['errors']}"
    )

    if run == 0:

        print("\nMatrice di confusione:")
        print("Colonne: predetto 0 | predetto 1 | pareggio")
        print("Righe:   reale 0, reale 1")

        print(results["confusion"])

        print(
            "Immagini valutate:",
            results["total"]
        )

print("\n=== FINE ===")



# =====================================================
# 8. ANALISI DEGLI ERRORI
# =====================================================

print("\n=== ANALISI ERRORI ===")

torch.manual_seed(0)

problems = []
offset = 0

with torch.no_grad():

    for images, labels in test_loader:

        spike_counts = forward_batch(images)

        predictions = spike_counts.argmax(dim=1)

        is_tie = spike_counts[:, 0] == spike_counts[:, 1]

        # Identifichiamo errori oppure pareggi
        is_problem = (predictions != labels) | is_tie

        problem_indices = torch.where(is_problem)[0]

        for i in problem_indices.tolist():

            problem = {
                "image": images[i].squeeze(0),
                "label": labels[i].item(),
                "prediction": predictions[i].item(),
                "spikes": spike_counts[i].tolist(),
                "tie": is_tie[i].item(),
                "mnist_index": test_indices[offset + i].item()
            }

            problems.append(problem)

        offset += labels.size(0)

print("Numero di casi problematici:", len(problems))

for p in problems:

    result = "PAREGGIO" if p["tie"] else "ERRORE"

    print(
        f"MNIST index={p['mnist_index']} | "
        f"Reale={p['label']} | "
        f"Predetta={p['prediction'] if not p['tie'] else 'indeterminata'} | "
        f"Spike={p['spikes']} | "
        f"{result}"
    )



if len(problems) > 0:

    ncols = min(3, len(problems))
    nrows = (len(problems) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False
    )

    # Nascondiamo tutti gli assi
    for ax in axes.flat:
        ax.axis("off")

    for ax, p in zip(axes.flat, problems):

        ax.imshow(
            p["image"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        result = "PAREGGIO" if p["tie"] else "ERRORE"

        ax.set_title(
            f"{result} | Reale: {p['label']}\n"
            f"Spike [0,1]: {p['spikes']}\n"
            f"MNIST index: {p['mnist_index']}",
            fontsize=10
        )

    plt.tight_layout()

    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/mnist_errors.png", dpi=200)

    plt.show()
