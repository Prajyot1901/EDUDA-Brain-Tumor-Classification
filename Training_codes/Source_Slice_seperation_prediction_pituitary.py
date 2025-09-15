import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def binary_threshold(image):
    """
    Applies binary thresholding to a grayscale image.
    Non-zero values → 255, zero remains 0.
    """
    binary_image = np.where(image > 35, 255, 0).astype(np.uint8)
    return binary_image


# -------------------------------
# Model definition (must match training)
# -------------------------------
class DilatedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DilatedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # -------------------------------
    # Paths
    # -------------------------------
    input_folder = r"Transfer learning\UDA_MRI\Datasets\Source\pituitary"

    # Define custom output folders for each class
    output_folders = {
        "Axial":    r"Transfer learning\UDA_MRI\Slice_Separated_Dataset\Axial\Source\pituitary",
        "Coronal":  r"Transfer learning\UDA_MRI\Slice_Separated_Dataset\Coronal\Source\pituitary",
        "Sagittal": r"Transfer learning\UDA_MRI\Slice_Separated_Dataset\Sagittal\Source\pituitary"
    }

    # Make sure all output directories exist
    for path in output_folders.values():
        os.makedirs(path, exist_ok=True)

    classes = ['Axial', 'Coronal', 'Sagittal']

    # -------------------------------
    # Device (GPU if available)
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # -------------------------------
    # Load model (DilatedCNN with trained weights)
    # -------------------------------
    net = DilatedCNN(num_classes=len(classes)).to(device)
    state = torch.load(
        r"Transfer learning\UDA_MRI\Model weights\Slice_seperation.pth",
        map_location=device
    )
    net.load_state_dict(state)
    net.eval()

    # -------------------------------
    # Transform (must match training)
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # -------------------------------
    # Process input images
    # -------------------------------
    with torch.no_grad():
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(input_folder, filename)

            # Load once as RGB for saving later
            original_img = Image.open(img_path).convert("RGB")

            # Make a grayscale copy for thresholding
            gray_np = np.array(original_img.convert("L"))
            binary_np = binary_threshold(gray_np)
            binary_img = Image.fromarray(binary_np).convert("RGB")

            # Apply transforms and move to device
            input_tensor = transform(binary_img).unsqueeze(0).to(device)

            # Prediction
            outputs = net(input_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = classes[predicted.item()]

            # Save original image into the chosen output folder
            save_path = os.path.join(output_folders[pred_class], filename)
            original_img.save(save_path)

            print(f"{filename} → Predicted: {pred_class}, saved to {save_path}")

    print("Classification & saving completed!")
