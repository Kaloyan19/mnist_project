import numpy as np
import matplotlib.pyplot as plt

mlp_losses = np.load("train_losses.npy")[:12]
mlp_acc = np.load("test_accuracies.npy")[:12]

cnn_losses = np.load("cnn_train_losses.npy")
cnn_acc = np.load("cnn_test_accuracies.npy")

epochs = np.arange(1, 13)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, mlp_losses, label='MLP Loss', color='blue', linestyle='--')
plt.plot(epochs, cnn_losses, label='CNN Loss', color='orange')
plt.title('Training Loss: MLP vs CNN')
plt.xlabel('Епоха')
plt.ylabel('Загуба')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, mlp_acc, label='MLP Accuracy', color='blue', linestyle='--')
plt.plot(epochs, cnn_acc, label='CNN Accuracy', color='orange')
plt.title('Test Accuracy: MLP vs CNN')
plt.xlabel('Епоха')
plt.ylabel('Точност (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("mlp_vs_cnn_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

print("Сравнителната графика е запазена като mlp_vs_cnn_comparison.png")
