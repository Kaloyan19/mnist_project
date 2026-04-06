import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import BetterMLP


model = BetterMLP()
model.load_state_dict(torch.load("models/mnist_mlp_model_2.pth", map_location=torch.device('cpu')))
model.eval()
print("Новият по-добър модел е зареден успешно!\n")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_digit(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.2)

    img_array = np.array(img)
    img_array[img_array < 90] = 0
    img_array[img_array >= 90] = 255
    img = Image.fromarray(img_array)

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


predict_digit("mnist_photos/3.jpg")
predict_digit("mnist_photos/1.1.png")
predict_digit("mnist_photos/3.1.png")
predict_digit("mnist_photos/7.png")
predict_digit("mnist_photos/6.jpg")
