import torch
from snntorch import spikegen
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Caricamento del dataset MNIST
mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Cerchiamo un esempio di 0 e uno di 1
examples = {}

for image, label in mnist_train:
    if label in (0, 1) and label not in examples:
        examples[label] = image

    if len(examples) == 2:
        break

torch.manual_seed(0)

image = examples[0]

num_steps = 30

spike_train = spikegen.rate(
    image,
    num_steps=num_steps
)

print("Shape immagine:", image.shape)
print("Shape spike train:", spike_train.shape)


fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Immagine originale")

axes[1].imshow(spike_train[0].squeeze(0), cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Spike al passo 1")

mean_spikes = spike_train.mean(dim=0)

axes[2].imshow(mean_spikes.squeeze(0), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Firing rate medio")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()