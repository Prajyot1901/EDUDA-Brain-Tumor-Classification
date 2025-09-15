import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# =======================================================
# CONFIG
# =======================================================
labels = ['glioma', 'meningioma', 'pituitary']
image_size = 224
num_classes = len(labels)
BATCH_SIZE = 32
EPOCHS = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================================================
# DATA LOADING
# =======================================================
def load_dataset(data_dir, labels, image_size):
    images, targets = [], []
    for label in labels:
        path = os.path.join(data_dir, label)
        for file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            image = cv2.imread(os.path.join(path, file), 0)
            image = cv2.bilateralFilter(image, 2, 50, 50)
            image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype(np.float32) / 255.0
            images.append(image)
            targets.append(labels.index(label))
    return np.array(images), np.array(targets)

MRI_1, MRI_1_label = load_dataset(
    r"UDA_MRI\Datasets\Source", labels, image_size
)
MRI_2, MRI_2_label = load_dataset(
    r"UDA_MRI\Datasets\Target", labels, image_size
)

MRI_1, MRI_1_label = shuffle(MRI_1, MRI_1_label, random_state=42)
MRI_1, MRI_1_val, MRI_1_label, MRI_1_label_val = train_test_split(
    MRI_1, MRI_1_label, test_size=0.2, random_state=42
)

# Convert to tensors
MRI_1_tensor = torch.tensor(MRI_1, dtype=torch.float32).permute(0, 3, 1, 2)
MRI_1_val_tensor = torch.tensor(MRI_1_val, dtype=torch.float32).permute(0, 3, 1, 2)
MRI_2_tensor = torch.tensor(MRI_2, dtype=torch.float32).permute(0, 3, 1, 2)
MRI_1_label_tensor = torch.tensor(MRI_1_label, dtype=torch.long)
MRI_1_label_val_tensor = torch.tensor(MRI_1_label_val, dtype=torch.long)
MRI_2_label_tensor = torch.tensor(MRI_2_label, dtype=torch.long)

# Dataset class
class MRIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

MRI_1_loader = DataLoader(MRIDataset(MRI_1_tensor, MRI_1_label_tensor), batch_size=BATCH_SIZE, shuffle=True)
MRI_1_val_loader = DataLoader(MRIDataset(MRI_1_val_tensor, MRI_1_label_val_tensor), batch_size=BATCH_SIZE)
MRI_2_loader = DataLoader(MRIDataset(MRI_2_tensor, MRI_2_label_tensor), batch_size=BATCH_SIZE, shuffle=True)

# =======================================================
# MODEL
# =======================================================
class BaseClassifier(nn.Module):
    def __init__(self, backbone, in_features, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        features = self.dropout(features)
        logits = self.fc(features)
        probs = torch.softmax(logits, dim=1)
        return probs, features

# ----- Pretrained Models -----
def get_resnet50(num_classes=3, dropout=0.4, pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return BaseClassifier(model, in_features, num_classes, dropout)

def get_densenet121(num_classes=3, dropout=0.4, pretrained=True):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier.in_features
    model.classifier = nn.Identity()
    return BaseClassifier(model, in_features, num_classes, dropout)

def get_efficientnet_b0(num_classes=3, dropout=0.4, pretrained=True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Identity()
    return BaseClassifier(model, in_features, num_classes, dropout)

def get_convnext_tiny(num_classes=3, dropout=0.4, pretrained=True):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Identity()
    return BaseClassifier(model, in_features, num_classes, dropout)

def get_vit_b16(num_classes=3, dropout=0.4, pretrained=True):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Identity()
    return BaseClassifier(model, in_features, num_classes, dropout)

all_models = {
    "ResNet50": get_resnet50(),
    "DenseNet121": get_densenet121(),
    "EfficientNetB0": get_efficientnet_b0(),
    "ConvNeXtTiny": get_convnext_tiny(),
    "ViT_B16": get_vit_b16(),
}

# =======================================================
# TRAINING + EVAL
# =======================================================
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            probs, _ = model(images)
            pred = torch.argmax(probs, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(targets.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro")
    rec = recall_score(labels, preds, average="macro")
    return acc, f1, prec, rec

def train_model(model, model_name, train_loader, val_loader, epochs=EPOCHS, lr=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_prec": [], "val_rec": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probs, _ = model(images)
            loss = F.cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(probs, 1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_acc, val_f1, val_prec, val_rec = evaluate_model(model, val_loader)

        # Save history
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_prec"].append(val_prec)
        history["val_rec"].append(val_rec)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(save_dir, f"{model_name}_best.pth"))

    # Save last model
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_last.pth"))

    return model, history

# =======================================================
# MAIN LOOP
# =======================================================
save_dir = r"UDA_MRI\Model weights"
os.makedirs(save_dir, exist_ok=True)

all_logs = {}

for name, model in all_models.items():
    print(f"\n==============================")
    print(f"Training {name} ...")
    trained_model, history = train_model(model, name, MRI_1_loader, MRI_1_val_loader, epochs=EPOCHS)

    print(f"Evaluating {name} on Target Domain (MRI_2) ...")
    acc, f1, prec, rec = evaluate_model(trained_model, MRI_2_loader)

    all_logs[name] = {
        "history": history,
        "target_results": {"Acc": acc, "F1": f1, "Precision": prec, "Recall": rec}
    }

    print(f"{name} → Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

# Save all logs in one file
log_path = os.path.join(r"UDA_MRI\Outputs", "SOTA_models_training_log.pth")
torch.save(all_logs, log_path)
print(f"\n===== Training Complete. Logs saved to {log_path} =====")
