import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# ==========================================================
# Paths
# ==========================================================
source_root = r"Transfer learning\UDA_MRI\Datasets\Source"
target_root = r"Transfer learning\UDA_MRI\Datasets\Target"
weights_path = r"Transfer learning\UDA_MRI\Model weights\best_dann_model.pth"
log_txt = r"Transfer learning\UDA_MRI\Outputs\DANN_eval_results.txt"
log_pth = r"Transfer learning\UDA_MRI\Outputs\DANN_eval_results.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# Data loading (same preprocessing as training)
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

source_ds = datasets.ImageFolder(source_root, transform=transform)
target_ds = datasets.ImageFolder(target_root, transform=transform)

source_loader = DataLoader(source_ds, batch_size=32, shuffle=False)
target_loader = DataLoader(target_ds, batch_size=32, shuffle=False)

# ==========================================================
# Model Definition (must match training exactly)
# ==========================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN(nn.Module):
    def __init__(self, num_classes=3):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(64 * 53 * 53, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(64 * 53 * 53, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        f = self.feature_extractor(x)
        f = f.view(f.size(0), -1)
        class_output = self.class_classifier(f)
        domain_output = self.domain_classifier(self.grl(f))
        return class_output, domain_output

# ==========================================================
# Load model weights
# ==========================================================
model = DANN(num_classes=len(source_ds.classes)).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print(f"Loaded weights from: {weights_path}")

# ==========================================================
# Evaluation Helper
# ==========================================================
def eval_domain(loader, name):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            preds = out.argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=loader.dataset.classes)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "report": report
    }

# ==========================================================
# Run evaluation
# ==========================================================
print("Evaluating Source domain...")
src_metrics = eval_domain(source_loader, "Source")
print("Evaluating Target domain...")
tgt_metrics = eval_domain(target_loader, "Target")


def print_metrics(name, m):
    print(f"\n{name} Domain Results")
    print(f"Accuracy:  {m['accuracy']*100:.2f}%")
    print(f"F1-macro:  {m['f1_macro']:.4f}")
    print(f"Precision: {m['precision_macro']:.4f}")
    print(f"Recall:    {m['recall_macro']:.4f}")
    print("Classification Report:")
    print(m['report'])

print_metrics("Source", src_metrics)
print_metrics("Target", tgt_metrics)


# ==========================================================
# Save results
# ==========================================================
with open(log_txt, "w") as f:
    f.write("=== DANN Evaluation Results ===\n\n")
    f.write("Source Domain:\n")
    f.write(f"Accuracy: {src_metrics['accuracy']*100:.2f}%\n")
    f.write(f"F1-macro: {src_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision-macro: {src_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall-macro: {src_metrics['recall_macro']:.4f}\n")
    f.write(src_metrics['report'] + "\n\n")

    f.write("Target Domain:\n")
    f.write(f"Accuracy: {tgt_metrics['accuracy']*100:.2f}%\n")
    f.write(f"F1-macro: {tgt_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision-macro: {tgt_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall-macro: {tgt_metrics['recall_macro']:.4f}\n")
    f.write(tgt_metrics['report'] + "\n")

torch.save({"Source": src_metrics, "Target": tgt_metrics}, log_pth)
print(f"\nResults saved to:\n  {log_txt}\n  {log_pth}")
