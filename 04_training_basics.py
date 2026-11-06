import torch
import torch.nn as nn

# Connessione con due ingressi e un'uscita
connection = nn.Linear(2, 1, bias=False)

# Pesi iniziali
with torch.no_grad():
    connection.weight.copy_(torch.tensor([[0.2, 0.2]]))

# Dati supervisionati
x = torch.tensor([1.0, 0.0])
target = torch.tensor([0.8])

# Funzione di errore
loss_fn = nn.MSELoss()

# Ottimizzatore
optimizer = torch.optim.SGD(
    connection.parameters(),
    lr=0.5
)


# Training
for epoch in range(5):

    optimizer.zero_grad()

    prediction = connection(x)

    loss = loss_fn(prediction, target)

    loss.backward()

    optimizer.step()

    print("Epoca:", epoch + 1)
    print("Prediction:", prediction.item())
    print("Loss:", loss.item())
    print("Pesi aggiornati:", connection.weight)