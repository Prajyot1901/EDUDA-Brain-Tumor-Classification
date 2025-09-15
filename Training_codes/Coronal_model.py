import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import logging

# -----------------------
# Logging Config
# -----------------------

logger = logging.getLogger()

# -----------------------
# CONFIG
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MRI1_ROOT = r"UDA_MRI\Slice_Separated_Dataset\Coronal_dataset\Source"
MRI2_ROOT = r"UDA_MRI\Slice_Separated_Dataset\Coronal_dataset\Target"

LABELS = ['glioma', 'meningioma', 'pituitary']
NUM_CLASSES = len(LABELS)
IMAGE_SIZE = 224
BATCH_SIZE = 16
PHASE_1_EPOCHS = 20
PHASE_2_EPOCHS = 100
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# -----------------------
# Dataset
# -----------------------
def load_dataset(folder_root, labels, image_size):
    images, targets = [], []
    for idx, label in enumerate(labels):
        folder = os.path.join(folder_root, label)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.bilateralFilter(img, 5, 75, 75)
            img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            img_color = cv2.resize(img_color, (image_size,image_size))
            img_color = img_color.astype(np.float32)/255.0
            img_color = (img_color - IMAGENET_MEAN)/IMAGENET_STD
            images.append(img_color)
            targets.append(idx)
    return np.array(images), np.array(targets)

class MRIDataset(Dataset):
    def __init__(self, images_np, labels_np):
        self.images = torch.tensor(images_np, dtype=torch.float32).permute(0,3,1,2)
        self.labels = torch.tensor(labels_np, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# -----------------------
# Model
# -----------------------
class ResNet50_FC(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, return_features=False):
        with torch.no_grad():
            feats = torch.flatten(self.feature_extractor(x),1)
        h1 = self.relu(self.fc1(feats))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        logits = self.fc4(h3)
        if return_features:
            return logits, h3
        return logits

# -----------------------
# Training / Evaluation
# -----------------------
def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def train_one_epoch(model, source_loader, target_loader, optimizer, criterion, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    target_iter = iter(target_loader)
    for imgs_s, labels_s in tqdm(source_loader, desc="Train batches"):
        try:
            imgs_t, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            imgs_t, _ = next(target_iter)

        imgs_s, labels_s, imgs_t = imgs_s.to(device), labels_s.to(device), imgs_t.to(device)

        optimizer.zero_grad()
        logits_s, feats_s = model(imgs_s, return_features=True)
        _, feats_t = model(imgs_t, return_features=True)

        loss_cls = criterion(logits_s, labels_s)
        loss_coral = compute_mmd_loss(feats_s, feats_t, multi_gaussian_kernel)
        loss = loss_cls + 1e-3*loss_coral*0 # disabled

        loss.backward()
        optimizer.step()

        batch_size = imgs_s.size(0)
        running_loss += loss.item()*batch_size
        running_acc += (torch.argmax(logits_s,1) == labels_s).float().sum().item()
        n += batch_size
    return running_loss/n, running_acc/n

def evaluate(model, loader, device):
    model.eval()
    accs, n = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            batch_size = imgs.size(0)
            accs += accuracy_from_logits(logits, labels)*batch_size
            n += batch_size
    return accs/n

# -----------------------
# Pseudo-labeling & DA
# -----------------------
def assign_pseudo_labels(model, loader, device):
    model.eval()
    all_imgs, pseudo_labels = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            mapped = []
            for p in preds:
                if p == 0: mapped.append(1)
                elif p == 1: mapped.append(0)
                else: mapped.append(2)
            all_imgs.append(imgs.cpu())
            pseudo_labels.append(np.array(mapped))
    
    all_imgs = torch.cat(all_imgs, dim=0)
    pseudo_labels = np.concatenate(pseudo_labels, axis=0)
    return all_imgs, pseudo_labels

def train_with_pseudo_labels(model, source_loader, target_loader, optimizer, criterion, device, lambda_coral=(1e-4)*5):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    target_iter = iter(target_loader)

    for imgs_s, labels_s in tqdm(source_loader, desc="DA Train batches"):
        try:
            imgs_t, labels_t = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            imgs_t, labels_t = next(target_iter)

        imgs_s, labels_s = imgs_s.to(device), labels_s.to(device)
        imgs_t, labels_t = imgs_t.to(device), labels_t.to(device)

        optimizer.zero_grad()
        logits_s, feats_s = model(imgs_s, return_features=True)
        logits_t, feats_t = model(imgs_t, return_features=True)

        loss_cls_s = criterion(logits_s, labels_s)
        loss_cls_t = criterion(logits_t, labels_t)
        loss_cls = loss_cls_s + loss_cls_t

        loss_coral = compute_mmd_loss(feats_s, feats_t, multi_gaussian_kernel)
        loss = loss_cls + lambda_coral * loss_coral
        loss.backward()
        optimizer.step()

        batch_size = imgs_s.size(0)
        running_loss += loss.item() * batch_size
        running_acc += (torch.argmax(logits_s, 1) == labels_s).float().sum().item()
        n += batch_size

    return running_loss / n, running_acc / n

# -----------------------
# Feature Extraction
# -----------------------
def extract_features_after_fc3(model, loader, device):
    model.eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, f = model(imgs, return_features=True)
            feats.append(f.cpu().numpy())
            labs.append(labels.numpy())
    feats = np.concatenate(feats, axis=0)
    labs = np.concatenate(labs, axis=0)
    return feats, labs

# -----------------------
# MMD Loss
# -----------------------
def multi_gaussian_kernel(X1, X2, sigmas=(1, 2, 4, 8, 16)):
    dist_sq = torch.cdist(X1, X2, p=2) ** 2
    kernels = [torch.exp(-dist_sq / (2.0 * sigma ** 2)) for sigma in sigmas]
    return sum(kernels) / len(kernels)

def compute_mmd_loss(Xs, Xt, kernel_fn=None):
    Kss = kernel_fn(Xs, Xs)
    Ktt = kernel_fn(Xt, Xt)
    Kst = kernel_fn(Xs, Xt)
    mmd_value = torch.mean(Kss) + torch.mean(Ktt) - 2.0 * torch.mean(Kst)
    return mmd_value

# -----------------------
# Reports
# -----------------------
def final_classification_report(model, loader, device, labels=LABELS):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    report = classification_report(all_labels, all_preds, target_names=labels, digits=4)
    logger.info("===== Final Classification Report on MRI-2 =====")
    logger.info("\n" + report)
    return report

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    mri1_images, mri1_labels = load_dataset(MRI1_ROOT, LABELS, IMAGE_SIZE)
    mri2_images, mri2_labels = load_dataset(MRI2_ROOT, LABELS, IMAGE_SIZE)

    mri1_images, mri1_labels = shuffle(mri1_images, mri1_labels, random_state=SEED)
    mri1_train, mri1_val, mri1_train_lbl, mri1_val_lbl = train_test_split(
        mri1_images, mri1_labels, test_size=0.2, random_state=SEED, stratify=mri1_labels
    )

    train_loader = DataLoader(MRIDataset(mri1_train, mri1_train_lbl), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(MRIDataset(mri1_val, mri1_val_lbl), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    mri1_loader  = DataLoader(MRIDataset(mri1_images, mri1_labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    mri2_loader  = DataLoader(MRIDataset(mri2_images, mri2_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = ResNet50_FC(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.fc1.parameters()) + 
                                 list(model.fc2.parameters()) + 
                                 list(model.fc3.parameters()) + 
                                 list(model.fc4.parameters()), lr=LR)

    best_score = 0.0
    best_model_path = r"UDA_MRI\Model weights\Coronal_weights.pth"

    # Phase 1
    logger.info("===== Phase 1: Training on MRI-1 =====")
    for epoch in range(1, PHASE_1_EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, mri2_loader, optimizer, criterion, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        logger.info(f"[Phase 1][Epoch {epoch}/{PHASE_1_EPOCHS}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Pseudo labels
    logger.info("===== Assigning Pseudo Labels to MRI-2 =====")
    pseudo_imgs, pseudo_labels = assign_pseudo_labels(model, mri2_loader, DEVICE)
    pseudo_dataset = torch.utils.data.TensorDataset(pseudo_imgs, torch.tensor(pseudo_labels, dtype=torch.long))
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # -----------------------
    # Phase 2 Logs
    # -----------------------
    phase2_logs = {
        "epoch_data": [],
        "best_model": {},
        "final_report": ""
    }

    # Phase 2 training
    logger.info("===== Phase 2: Domain Adaptation Training =====")
    for epoch in range(1, PHASE_2_EPOCHS+1):
        train_loss, train_acc = train_with_pseudo_labels(model, train_loader, pseudo_loader, optimizer, criterion, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        test_acc = evaluate(model, mri2_loader, DEVICE)

        phase2_logs["epoch_data"].append({
            "epoch": epoch,
            "loss": train_loss,
            "val_acc": val_acc,
            "test_acc": test_acc
        })

        logger.info(f"[Phase 2][Epoch {epoch}/{PHASE_2_EPOCHS}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

        score = val_acc * test_acc
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_model_path)
            phase2_logs["best_model"] = {
                "epoch": epoch,
                "loss": train_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "score": score
            }
            logger.info(f"*** Best model updated at epoch {epoch}, score={best_score:.4f} ***")

    # Load best
    model.load_state_dict(torch.load(best_model_path))
    logger.info("Loaded best model for final evaluation.")

    # Final classification report
    final_report = final_classification_report(model, mri2_loader, DEVICE)
    phase2_logs["final_report"] = final_report

    # Save logs
    log_save_path = r"UDA_MRI\Outputs\Coronal_training.pth"
    torch.save(phase2_logs, log_save_path)
    logger.info(f"Phase 2 logs saved to {log_save_path}")

# -----------------------
# Helper: Reload & Inspect Logs
# -----------------------
def load_logs(log_file):
    logs = torch.load(log_file)
    print("\n=== Training History (last 5 epochs) ===")
    for entry in logs["epoch_data"][-5:]:
        print(entry)
    print("\n=== Best Model ===")
    print(logs["best_model"])
    print("\n=== Final Classification Report ===")
    print(logs["final_report"])
    return logs
