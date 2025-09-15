import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# -----------------------
# Config
# -----------------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

MRI1_ROOT = r"Transfer learning\UDA_MRI\Datasets\Source"
MRI2_ROOT = r"Transfer learning\UDA_MRI\Datasets\Target"
BEST_MODEL_PATH = r"Transfer learning\UDA_MRI\Model weights\DDC_model_weights.pth"
RESULT_LOG_PATH = r"Transfer learning\UDA_MRI\Outputs\DDC_eval_results.pth"

LABELS = ['glioma','meningioma','pituitary']
NUM_CLASSES = len(LABELS)
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -----------------------
# Dataset helpers
# -----------------------
def load_dataset(folder_root, labels, image_size):
    imgs, tgts = [], []
    for idx, lbl in enumerate(labels):
        folder = os.path.join(folder_root, lbl)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.bilateralFilter(img, 5, 75, 75)
            img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_size,image_size))
            img = img.astype(np.float32)/255.0
            img = (img - IMAGENET_MEAN)/IMAGENET_STD
            imgs.append(img)
            tgts.append(idx)
    return np.array(imgs), np.array(tgts)

class MRIDataset(Dataset):
    def __init__(self, images_np, labels_np):
        self.images = torch.tensor(images_np, dtype=torch.float32).permute(0,3,1,2)
        self.labels = torch.tensor(labels_np, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# -----------------------
# Model (same as training)
# -----------------------
from torchvision import models
class ResNet50_FC(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        with torch.no_grad():
            feats = torch.flatten(self.feature_extractor(x),1)
        x = self.relu(self.fc1(feats))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=LABELS, digits=4)
    return acc, cm, report

# -----------------------
# Main evaluation
# -----------------------
if __name__ == "__main__":
    # Load datasets with same seed & stratification as training
    src_imgs, src_lbls = load_dataset(MRI1_ROOT, LABELS, IMAGE_SIZE)
    tgt_imgs, tgt_lbls = load_dataset(MRI2_ROOT, LABELS, IMAGE_SIZE)

    src_imgs, src_lbls = shuffle(src_imgs, src_lbls, random_state=SEED)
    src_train, src_val, src_train_lbl, src_val_lbl = train_test_split(
        src_imgs, src_lbls, test_size=0.2, random_state=SEED, stratify=src_lbls
    )

    train_loader = DataLoader(MRIDataset(src_train, src_train_lbl), batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(MRIDataset(src_val, src_val_lbl), batch_size=BATCH_SIZE, shuffle=False)
    tgt_loader   = DataLoader(MRIDataset(tgt_imgs, tgt_lbls), batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = ResNet50_FC(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = {}

    for name, loader in [("source_val",   val_loader),
                         ("target_full",  tgt_loader)]:
        acc, cm, report = evaluate(model, loader)
        print(f"\n==== {name.upper()} ====")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print(report)
        results[name] = {
            "accuracy": float(acc),
            "confusion_matrix": cm,
            "classification_report": report
        }
        # Optional: quick heatmap
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.show()

    torch.save(results, RESULT_LOG_PATH)
    print(f"\nAll evaluation results saved to: {RESULT_LOG_PATH}")
