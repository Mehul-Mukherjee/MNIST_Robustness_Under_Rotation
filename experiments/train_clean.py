import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# DATA (CLEAN MNIST)
# --------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# --------------------
# MODEL
# --------------------
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# --------------------
# TRAIN LOOP
# --------------------
for epoch in range(3):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --------------------
# SAVE MODEL
# --------------------
torch.save(model.state_dict(), "results/clean_model.pth")
print("Saved clean model")