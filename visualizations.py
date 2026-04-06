import numpy as np
import matplotlib.pyplot as plt

train_losses = np.load("train_losses.npy")
test_accuracies = np.load("test_accuracies.npy")

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

print("Графиките са запазени като loss_accuracy_curves.png")
