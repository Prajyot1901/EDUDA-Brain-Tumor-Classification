import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as models
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# CONFIG
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_PATH = r".\Model weights\Sagittal_weights.pth"
MRI1_ROOT       = r".\Slice_Separated_Dataset\Sagittal\Source"
MRI2_ROOT       = r".\Slice_Separated_Dataset\Sagittal\Target"
LOG_SAVE_PATH   = r".\Outputs"
os.makedirs(LOG_SAVE_PATH, exist_ok=True)

LABELS = ['glioma', 'meningioma', 'pituitary']
IMAGE_SIZE = 224
BATCH_SIZE = 16
SEED = 42

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -----------------------
# Dataset Utilities
# -----------------------
def load_dataset(folder_root, labels, image_size):
    images, targets = [], []
    for idx, label in enumerate(labels):
        folder = os.path.join(folder_root, label)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.bilateralFilter(img, 5, 75, 75)
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_color = cv2.resize(img_color, (image_size, image_size))
            img_color = img_color.astype(np.float32) / 255.0
            img_color = (img_color - IMAGENET_MEAN) / IMAGENET_STD
            images.append(img_color)
            targets.append(idx)
    return np.array(images), np.array(targets)

class MRIDataset(Dataset):
    def __init__(self, images_np, labels_np):
        self.images = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels_np, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# -----------------------
# Model
# -----------------------
class ResNet50_FC(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            feats = torch.flatten(self.feature_extractor(x), 1)
        x = self.relu(self.fc1(feats))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def evaluate(loader, model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            labels.extend(y.numpy())
    preds = np.array(preds); labels = np.array(labels)
    acc = (preds == labels).mean() * 100.0
    return acc, preds, labels

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Load Source (MRI-1) and Target (MRI-2)
    mri1_images, mri1_labels = load_dataset(MRI1_ROOT, LABELS, IMAGE_SIZE)
    mri2_images, mri2_labels = load_dataset(MRI2_ROOT, LABELS, IMAGE_SIZE)

    # --- Recreate the same source split as training ---
    mri1_images, mri1_labels = shuffle(mri1_images, mri1_labels, random_state=SEED)
    mri1_train, mri1_val, mri1_train_lbl, mri1_val_lbl = train_test_split(
        mri1_images, mri1_labels, test_size=0.2, random_state=SEED, stratify=mri1_labels
    )

    source_train_loader = DataLoader(MRIDataset(mri1_train, mri1_train_lbl),
                                     batch_size=BATCH_SIZE, shuffle=False)
    source_val_loader   = DataLoader(MRIDataset(mri1_val, mri1_val_lbl),
                                     batch_size=BATCH_SIZE, shuffle=False)
    target_loader       = DataLoader(MRIDataset(mri2_images, mri2_labels),
                                     batch_size=BATCH_SIZE, shuffle=False)

    # Load trained model
    model = ResNet50_FC(num_classes=len(LABELS)).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Evaluate ---
    src_train_acc, _, _ = evaluate(source_train_loader, model)
    src_val_acc,   _, _ = evaluate(source_val_loader,   model)
    tgt_acc, tgt_preds, tgt_labels = evaluate(target_loader, model)

    # Reports for target domain
    tgt_report = classification_report(tgt_labels, tgt_preds,
                                       target_names=LABELS, digits=4)
    tgt_cm = confusion_matrix(tgt_labels, tgt_preds)

    # Console Output
    print(f"\nSource Train Accuracy: {src_train_acc:.2f}%")
    print(f"Source Val Accuracy  : {src_val_acc:.2f}%")
    print(f"Target Test Accuracy : {tgt_acc:.2f}%")
    print("\nTarget Classification Report:\n", tgt_report)
    print("Target Confusion Matrix:\n", tgt_cm)
    
    
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        tgt_cm,
        annot=True,          # show numbers
        fmt="d",             # integer format
        cmap="Blues",        # color map
        xticklabels=LABELS,  # predicted class names
        yticklabels=LABELS   # true class names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Target Domain Confusion Matrix")
    plt.tight_layout()
    plt.show()
    
    
    

    # Save to .pth log
    log_dict = {
        "source_train_acc": src_train_acc,
        "source_val_acc": src_val_acc,
        "target_test_acc": tgt_acc,
        "target_classification_report": tgt_report,
        "target_confusion_matrix": tgt_cm,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    log_file = os.path.join(
        LOG_SAVE_PATH,
        f"sagittal_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    )
    torch.save(log_dict, log_file)
    print(f"\nResults saved to: {log_file}")

