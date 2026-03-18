import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ====================== ЗАРЕЖДАНЕ НА ЗАПАЗЕНИТЕ ДАННИ ======================
train_losses = np.load("train_losses.npy")
test_accuracies = np.load("test_accuracies.npy")

print(f"Train Losses shape: {train_losses.shape}")
print(f"Test Accuracies shape: {test_accuracies.shape}")

# ====================== 1. ГРАФИКА Loss & Accuracy ======================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='green')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_accuracy_curves.png", dpi=300)
plt.show()


# ====================== 2. CONFUSION MATRIX ======================
# Зареждане на модела
class BetterMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = BetterMLP()
model.load_state_dict(torch.load("mnist_mlp_model_2.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.ToTensor()
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Събиране на предсказания
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("Визуализациите са запазени като:")
print("- loss_accuracy_curves.png")
print("- confusion_matrix.png")
