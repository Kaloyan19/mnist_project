import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 1. Define the same model architecture
class BetterMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 2. Load the model
model = BetterMLP()
model.load_state_dict(torch.load("mnist_mlp_model.pth", map_location=torch.device('cpu')))
model.eval()

# 3. The "MNIST-ifier" Transform
# This takes ANY image and turns it into a 28x28 grayscale tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.6, 0.6)),  # Adds that softness
    transforms.ToTensor(),
    # Normalization is key: MNIST mean is 0.1307, std is 0.3081
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_my_digit(image_path):
    # Open image
    img = Image.open(image_path).convert('L')

    # IMPORTANT: If your photo is black ink on white paper, we MUST invert it.
    # MNIST is white ink on black background.
    # Check the middle pixel; if it's bright, we assume it needs inverting.
    if img.getpixel((5, 5)) > 127:
        img = ImageOps.invert(img)

    # Apply transforms
    img_tensor = transform(img).unsqueeze(0)

    # Show what the model actually "sees" (28x28 version)
    plt.imshow(img_tensor.squeeze(), cmap='gray')
    plt.title("What the Model Sees (28x28 Normalized)")
    plt.show()

    with torch.no_grad():
        output = model(img_tensor)
        predicted = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1)[0]
        confidence = prob[predicted].item() * 100

    print(f"File: {image_path} | Prediction: {predicted} | Confidence: {confidence:.2f}%")


# Run it
predict_my_digit("6.hand.png")