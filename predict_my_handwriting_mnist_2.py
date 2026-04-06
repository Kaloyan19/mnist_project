import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import BetterMLP

model = BetterMLP()
model.load_state_dict(torch.load("models/mnist_mlp_model_2.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.6, 0.6)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_my_digit(image_path):
    img = Image.open(image_path).convert('L')

    if img.getpixel((5, 5)) > 127:
        img = ImageOps.invert(img)

    img_tensor = transform(img).unsqueeze(0)

    plt.imshow(img_tensor.squeeze(), cmap='gray')
    plt.title("What the Model Sees (28x28 Normalized)")
    plt.show()

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1)[0]
        confidence = prob[predicted].item() * 100

    print(f"File: {image_path} | Prediction: {predicted} | Confidence: {confidence:.2f}%")


predict_my_digit("my_handwritten_edited_photos/6_hand.png")
predict_my_digit("my_handwritten_edited_photos/7_hand.png")
