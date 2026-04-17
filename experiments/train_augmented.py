import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import CNN
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# CUSTOM AUGMENTATION
# --------------------
class AddNoise:
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * 0.1
        return tensor + noise


transform = transforms.Compose([
    transforms.RandomRotation(30),   # 🔥 key part (your idea)
    transforms.ToTensor(),
    AddNoise()
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

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

    print(f"[AUG] Epoch {epoch+1}, Loss: {total_loss:.4f}")

# --------------------
# SAVE MODEL
# --------------------
torch.save(model.state_dict(), "results/augmented_model.pth")
print("Saved augmented model")