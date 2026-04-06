import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import BetterMLP
import torch


model = BetterMLP()
model.load_state_dict(torch.load("models/mnist_mlp_model_2.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.ToTensor()
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

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

print("\n=== ПРИМЕРИ С ГРЕШНО ПРЕДСКАЗАНИ ЦИФРИ ===\n")

model.eval()
count = 0
fig = plt.figure(figsize=(12, 8))

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(len(labels)):
            if predicted[i] != labels[i] and count < 12:
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
