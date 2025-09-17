import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# =====================================================
# CONFIG
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r".\Model weights\Slice_seperation.pth"  
SOURCE_ROOT = r".\Slice_Separation_Training_Testing\Testing_Source_binary"  
TARGET_ROOT = r".\Slice_Separation_Training_Testing\Testing_Target_binary"  
LOG_SAVE_PATH = r".\Outputs"         
os.makedirs(LOG_SAVE_PATH, exist_ok=True)

BATCH_SIZE = 4
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =====================================================
# Dataset loaders
# =====================================================
source_set = torchvision.datasets.ImageFolder(root=SOURCE_ROOT, transform=transform)
target_set = torchvision.datasets.ImageFolder(root=TARGET_ROOT, transform=transform)

source_loader = torch.utils.data.DataLoader(source_set, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

CLASSES = source_set.classes  # assumes same classes for both
print("Classes:", CLASSES)


# =====================================================
# Model Definition (must match training)
# =====================================================
class DilatedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DilatedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 2, dilation=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 4, dilation=4)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
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
        return self.fc2(x)


# =====================================================
# Evaluation helper
# =====================================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            total += labs.size(0)
            correct += (preds == labs).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    acc = 100.0 * correct / total
    return acc, all_preds, all_labels


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    # Load model
    model = DilatedCNN(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded from:", MODEL_PATH)

    # Evaluate on Source and Target
    src_acc, src_preds, src_labels = evaluate(model, source_loader)
    tgt_acc, tgt_preds, tgt_labels = evaluate(model, target_loader)

    print(f"Source Accuracy: {src_acc:.2f}%")
    print(f"Target Accuracy: {tgt_acc:.2f}%")

    # Save combined log as .pth
    log_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "classes": CLASSES,
        "source": {
            "accuracy": src_acc,
            "predictions": src_preds,
            "labels": src_labels
        },
        "target": {
            "accuracy": tgt_acc,
            "predictions": tgt_preds,
            "labels": tgt_labels
        }
    }

    log_file = os.path.join(
        LOG_SAVE_PATH,
        f"slice_separation_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    torch.save(log_dict, log_file)
    print("Results saved to:", log_file)

