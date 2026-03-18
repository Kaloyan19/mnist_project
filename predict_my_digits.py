import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# ====================== НОВИЯТ ПО-ДОБЪР МОДЕЛ ======================
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


# Зареждане на новия модел
model = BetterMLP()
model.load_state_dict(torch.load("mnist_mlp_model.pth", map_location=torch.device('cpu')))
model.eval()
print("Новият по-добър модел е зареден успешно!\n")

# ====================== ТРАНСФОРМАЦИЯ ======================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_digit(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.2)

    # Лек threshold
    img_array = np.array(img)
    img_array[img_array < 90] = 0
    img_array[img_array >= 90] = 255
    img = Image.fromarray(img_array)

    # Показваме
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.title(f"Вход за модела: {image_path}")
    plt.axis('off')
    plt.show()

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        predicted = torch.argmax(prob, dim=1).item()
        confidence = prob[0][predicted].item() * 100

    print(f"{image_path}  →  Предсказана цифра: {predicted}   |   Увереност: {confidence:.2f}%")
    print("-" * 60)


# ====================== ТЕСТ ======================
predict_digit("3.jpg")  # 7
predict_digit("1.1.png")  # 1
predict_digit("3.1.png")  # 3
predict_digit("7.png")  # 7
predict_digit("6.jpg")
predict_digit("6.hand.png")# 6 # 6
