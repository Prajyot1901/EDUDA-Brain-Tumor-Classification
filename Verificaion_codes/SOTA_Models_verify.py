import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torchvision.models as models
import timm

# =======================================================
# CONFIG
# =======================================================
labels = ['glioma', 'meningioma', 'pituitary']
image_size = 224
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- paths (adjust if needed) ---
src_dir = r".\Datasets\Source"
tgt_dir = r".\Datasets\Target"
weights_dir = r".\Model weights"
log_path = r".\Outputs\SOTA_models_eval_results.pth"

# =======================================================
# DATA LOADING (same preprocessing as training)
# =======================================================
def load_dataset(data_dir):
    imgs, lbls = [], []
    for label in labels:
        path = os.path.join(data_dir, label)
        for f in tqdm(os.listdir(path), desc=f"Loading {label}"):
            img = cv2.imread(os.path.join(path, f), 0)
            img = cv2.bilateralFilter(img, 2, 50, 50)
            img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            img = cv2.resize(img, (image_size, image_size))
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
            lbls.append(labels.index(label))
    return np.array(imgs), np.array(lbls)

def make_loader(images, labels):
    t_img = torch.tensor(images, dtype=torch.float32).permute(0,3,1,2)
    t_lbl = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(t_img, t_lbl)
    return DataLoader(ds, batch_size=BATCH_SIZE)

MRI_1, MRI_1_lbl = load_dataset(src_dir)
MRI_2, MRI_2_lbl = load_dataset(tgt_dir)
MRI_1_loader = make_loader(MRI_1, MRI_1_lbl)
MRI_2_loader = make_loader(MRI_2, MRI_2_lbl)

# =======================================================
# MODEL DEFINITION (match training exactly)
# =======================================================
class BaseClassifier(nn.Module):
    def __init__(self, backbone, in_features, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = self.dropout(feats)
        return torch.softmax(self.fc(feats), dim=1)

def get_resnet50():
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Identity()
    return BaseClassifier(m, in_f)

def get_densenet121():
    m = models.densenet121(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Identity()
    return BaseClassifier(m, in_f)

def get_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    return BaseClassifier(m, in_f)

def get_convnext_tiny():
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Identity()
    return BaseClassifier(m, in_f)

def get_vit_b16():
    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    in_f = m.head.in_features
    m.head = nn.Identity()
    return BaseClassifier(m, in_f)

model_builders = {
    "ResNet50": get_resnet50,
    "DenseNet121": get_densenet121,
    "EfficientNetB0": get_efficientnet_b0,
    "ConvNeXtTiny": get_convnext_tiny,
    "ViT_B16": get_vit_b16,
}

# =======================================================
# EVALUATION
# =======================================================
def evaluate(model, loader):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labs.extend(y.numpy())
    return {
        "Acc": accuracy_score(labs, preds),
        "F1": f1_score(labs, preds, average="macro"),
        "Precision": precision_score(labs, preds, average="macro"),
        "Recall": recall_score(labs, preds, average="macro"),
    }

# =======================================================
# MAIN LOOP
# =======================================================
results = {}

for name, builder in model_builders.items():
    ckpt = os.path.join(weights_dir, f"{name}_best.pth")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(weights_dir, f"{name}_last.pth")
    if not os.path.isfile(ckpt):
        print(f"[WARN] No checkpoint for {name}")
        continue

    print(f"\nEvaluating {name} from {ckpt}")
    model = builder().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    src_metrics = evaluate(model, MRI_1_loader)
    tgt_metrics = evaluate(model, MRI_2_loader)
    results[name] = {"MRI_1": src_metrics, "MRI_2": tgt_metrics}

    print(f"{name} → Source Acc: {src_metrics['Acc']:.4f} | Target Acc: {tgt_metrics['Acc']:.4f}")

torch.save(results, log_path)
print(f"\nEvaluation complete. Results saved to: {log_path}")

