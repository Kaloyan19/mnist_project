import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ====================== ЗАРЕЖДАНЕ НА МОДЕЛА ======================
class BetterMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 10)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
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

# ====================== CONFUSION MATRIX ======================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix - BetterMLP Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== ГРЕШНИ ПРИМЕРИ (много важен за доклада) ======================
print("\n=== ПРИМЕРИ С ГРЕШНО ПРЕДСКАЗАНИ ЦИФРИ ===\n")

model.eval()
count = 0
fig = plt.figure(figsize=(12, 8))

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            if predicted[i] != labels[i] and count < 12:  # показваме максимум 12 грешки
                count += 1
                plt.subplot(3, 4, count)
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title(f"True: {labels[i]} | Pred: {predicted[i]}")
                plt.axis('off')

                if count >= 12:
                    break
        if count >= 12:
            break

plt.tight_layout()
plt.savefig("error_examples.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Показани са {count} примера с грешки.")
print("Файловете са запазени:")
print("- confusion_matrix.png")
print("- error_examples.png")