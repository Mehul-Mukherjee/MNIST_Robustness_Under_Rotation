import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# LOAD TEST DATA
# -------------------------
transform = transforms.ToTensor()

test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)


# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()

    return correct / total


# -------------------------
# LOAD MODELS
# -------------------------
clean_model = CNN().to(device)
clean_model.load_state_dict(torch.load("results/clean_model.pth", map_location=device))

aug_model = CNN().to(device)
aug_model.load_state_dict(torch.load("results/augmented_model.pth", map_location=device))


# -------------------------
# ROTATION SWEEP
# -------------------------
angles = [0, 15, 30, 45, 60, 90]

clean_acc = []
aug_acc = []

for angle in angles:
    transform = transforms.Compose([
        transforms.RandomRotation((angle, angle)),
        transforms.ToTensor()
    ])

    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1000)

    clean_acc.append(evaluate(clean_model, test_loader))
    aug_acc.append(evaluate(aug_model, test_loader))

    print(f"Angle {angle}: Clean={clean_acc[-1]:.3f}, Aug={aug_acc[-1]:.3f}")


# -------------------------
# PLOT ROBUSTNESS CURVE
# -------------------------
plt.plot(angles, clean_acc, label="Clean Model")
plt.plot(angles, aug_acc, label="Augmented Model")

plt.xlabel("Rotation Angle")
plt.ylabel("Accuracy")
plt.title("CNN Robustness vs Rotation")
plt.legend()

plt.savefig("results/robustness_curve.png")
plt.show()