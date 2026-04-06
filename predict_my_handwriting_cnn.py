import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


model = SimpleCNN()
model.load_state_dict(torch.load("models/mnist_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

print("CNN моделът е зареден успешно (99.19% точност)")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.6, 0.6)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_my_digit(image_path):
    img = Image.open(image_path).convert('L')

    img = ImageEnhance.Contrast(img).enhance(2.5)

    img_array = np.array(img)
    coords = np.argwhere(img_array > 50)
    if len(coords) > 0:
        cy, cx = coords.mean(axis=0)
        shift_y = int(14 - cy)
        shift_x = int(14 - cx)

        new_array = np.zeros((28, 28), dtype=np.uint8)
        h, w = img_array.shape
        y_start = max(0, shift_y)
        x_start = max(0, shift_x)
        y_end = min(28, h + shift_y)
        x_end = min(28, w + shift_x)

        new_array[y_start:y_end, x_start:x_end] = img_array[
                                                  max(0, -shift_y):min(h, 28 - shift_y),
                                                  max(0, -shift_x):min(w, 28 - shift_x)
                                                  ]
        img = Image.fromarray(new_array)

    img_tensor = transform(img).unsqueeze(0)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_tensor.squeeze(), cmap='gray')
    plt.title("Какво вижда CNN (центрирана 28×28)")
    plt.axis('off')
    plt.show()

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item() * 100

    print(f"Снимка: {image_path}")
    print(f"Предсказана цифра: {predicted}")
    print(f"Увереност: {confidence:.2f}%")
    print("-" * 60)


predict_my_digit("my_handwritten_edited_photos/6_hand.png")
predict_my_digit("my_handwritten_edited_photos/7_hand.png")
