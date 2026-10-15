from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model_best.pth"

# ***需要预测的图片路径（修改这里）***
IMAGE_PATH = PROJECT_DIR / ".." / ".." / "images" / "test.jpg"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
CLASS_NAMES = ["Cat", "Dog"]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    
def load_model(model_path):
    model = CNN().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    with Image.open(image_path) as img:
        original_image = img.convert("RGB")

    processed_image = transform(original_image)

    input_tensor = processed_image.unsqueeze(0).to(DEVICE)

    return original_image, processed_image, input_tensor


def predict(model, input_tensor):
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        probabilities = probabilities[0].cpu()
        predicted_index = probabilities.argmax().item()

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_class, confidence, probabilities


def show_result(
    original_image,
    processed_image,
    predicted_class,
    confidence
):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        processed_image.permute(1, 2, 0)
    )

    axes[1].set_title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )
    axes[1].axis("off")

    plt.show()


def main():
    print("运行设备：", DEVICE)
    model = load_model(MODEL_PATH)

    original, processed, input_tensor = preprocess_image(
        IMAGE_PATH
    )

    predicted_class, confidence, probabilities = predict(
        model,
        input_tensor
    )

    print("\n猫狗分类结果:")
    print("图片路径：", IMAGE_PATH)
    print("预测类别：", predicted_class)
    print(f"预测置信度：{confidence}")
    print(f"Cat概率：{probabilities[0].item()}")
    print(f"Dog概率：{probabilities[1].item()}")

    show_result(
        original,
        processed,
        predicted_class,
        confidence
    )


if __name__ == "__main__":
    main()
