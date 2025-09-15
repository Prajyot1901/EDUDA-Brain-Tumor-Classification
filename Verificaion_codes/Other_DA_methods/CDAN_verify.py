import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# ==========================================================
# Paths
# ==========================================================
source_root = r"Transfer learning\UDA_MRI\Datasets\Source"
target_root = r"Transfer learning\UDA_MRI\Datasets\Target"
weights_path = r"Transfer learning\UDA_MRI\Model weights\cdan_model_weights.pth"
log_txt = r"Transfer learning\UDA_MRI\Outputs\cdan_eval_results.txt"
log_pth = r"Transfer learning\UDA_MRI\Outputs\cdan_eval_results.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# Model definitions (match training code)
# ==========================================================
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=64*53*53, num_classes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=64*53*53*3, hidden_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.fc(x)

# ==========================================================
# Data loading
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
# Load model weights
# ==========================================================
feature_extractor = FeatureExtractor().to(device)
classifier = Classifier(num_classes=len(source_ds.classes)).to(device)
domain_discriminator = DomainDiscriminator().to(device)

checkpoint = torch.load(weights_path, map_location=device)
feature_extractor.load_state_dict(checkpoint['feature_extractor'])
classifier.load_state_dict(checkpoint['classifier'])
domain_discriminator.load_state_dict(checkpoint['domain_discriminator'])

feature_extractor.eval()
classifier.eval()
print(f"Loaded CDAN weights from: {weights_path}")

# ==========================================================
# Evaluation helper
# ==========================================================
def evaluate(loader, dataset_name):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = feature_extractor(imgs)
            logits = classifier(feats.view(feats.size(0), -1))
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro')
    rec  = recall_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds,
                                   target_names=loader.dataset.classes)

    print(f"\n=== {dataset_name} Evaluation ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"F1-macro: {f1:.4f}")
    print(f"Precision-macro: {prec:.4f}")
    print(f"Recall-macro   : {rec:.4f}")
    print("Classification Report:\n", report)
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
src_metrics = evaluate(source_loader, "Source Domain")
tgt_metrics = evaluate(target_loader, "Target Domain")

# ==========================================================
# Save logs
# ==========================================================
os.makedirs(os.path.dirname(log_txt), exist_ok=True)

with open(log_txt, "w") as f:
    f.write("=== CDAN Evaluation Results ===\n\n")
    f.write("Source Domain:\n")
    f.write(f"Accuracy : {src_metrics['accuracy']*100:.2f}%\n")
    f.write(f"F1-macro: {src_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision-macro: {src_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall-macro: {src_metrics['recall_macro']:.4f}\n")
    f.write(src_metrics['report'] + "\n\n")

    f.write("Target Domain:\n")
    f.write(f"Accuracy : {tgt_metrics['accuracy']*100:.2f}%\n")
    f.write(f"F1-macro: {tgt_metrics['f1_macro']:.4f}\n")
    f.write(f"Precision-macro: {tgt_metrics['precision_macro']:.4f}\n")
    f.write(f"Recall-macro: {tgt_metrics['recall_macro']:.4f}\n")
    f.write(tgt_metrics['report'] + "\n")

torch.save({"Source": src_metrics, "Target": tgt_metrics}, log_pth)

print(f"\nResults saved to:\n  {log_txt}\n  {log_pth}")
