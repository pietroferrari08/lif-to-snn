
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Caricamento del dataset MNIST
mnist_train = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print("Numero di immagini:", len(mnist_train))

# Cerchiamo un esempio di 0 e uno di 1
examples = {}

for image, label in mnist_train:
    if label in (0, 1) and label not in examples:
        examples[label] = image

    if len(examples) == 2:
        break

# Analizziamo e visualizziamo le immagini
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

for digit in (0, 1):
    image = examples[digit]

    print(f"Cifra {digit}")
    print("Shape:", image.shape)
    print("Intensità minima:", image.min().item())
    print("Intensità massima:", image.max().item())

    axes[digit].imshow(image.squeeze(0), cmap="gray")
    axes[digit].set_title(f"Cifra {digit}")
    axes[digit].axis("off")

plt.tight_layout()
plt.show()
