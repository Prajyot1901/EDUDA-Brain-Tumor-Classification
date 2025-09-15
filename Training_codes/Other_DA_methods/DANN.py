import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Function
from sklearn.metrics import classification_report
import numpy as np

# ---------------------------
# Gradient Reversal Layer
# ---------------------------
class GradientReversalFunction(Function):
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


# ---------------------------
# Model Definition
# ---------------------------
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
            nn.Linear(64 * 53 * 53, 100),  # adjust input size to your image dims
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(64 * 53 * 53, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # 2 domains: source and target
        )

        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output


# ---------------------------
# Data Loading
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

source_dataset = datasets.ImageFolder(
    root=r"UDA_MRI\Datasets\Source",
    transform=transform
)
target_dataset = datasets.ImageFolder(
    root=r"UDA_MRI\Datasets\Target",
    transform=transform
)

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)


# ---------------------------
# Training Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DANN(num_classes=3).to(device)

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0.0
log_file = open(r"UDA_MRI\Outputs\DANN_log.txt", "w")

# store metrics for .pth file
metrics = {
    "loss": [],
    "cls_loss": [],
    "dom_loss": [],
    "src_acc": [],
    "tgt_acc": []
}


# ---------------------------
# Evaluation Helper
# ---------------------------
def evaluate(loader, name="Target", print_report=True):
    model.eval()
    correct, total = 0, 0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            class_output, _ = model(data)
            preds = class_output.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = 100 * correct / total
    if print_report:
        print(f"{name} Domain Accuracy: {acc:.2f}%")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=loader.dataset.classes))
    return acc


# ---------------------------
# Training Loop
# ---------------------------
def train(num_epochs=20):
    global best_acc
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls, total_dom = 0, 0, 0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        n_batches = min(len(source_iter), len(target_iter))

        for i in range(n_batches):
            source_data, source_label = next(source_iter)
            target_data, _ = next(target_iter)

            source_data, source_label = source_data.to(device), source_label.to(device)
            target_data = target_data.to(device)

            p = float(i + epoch * n_batches) / (num_epochs * n_batches)
            alpha = 2. / (1. + torch.exp(torch.tensor(-10 * p))) - 1

            # Source forward
            class_output, domain_output = model(source_data, alpha)
            cls_loss = criterion_class(class_output, source_label)
            dom_loss_src = criterion_domain(domain_output, torch.zeros(len(source_data), dtype=torch.long).to(device))

            # Target forward
            _, domain_output = model(target_data, alpha)
            dom_loss_tgt = criterion_domain(domain_output, torch.ones(len(target_data), dtype=torch.long).to(device))

            # Combine losses
            loss = cls_loss + dom_loss_src + dom_loss_tgt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_dom += (dom_loss_src.item() + dom_loss_tgt.item())

        # Evaluate at the end of each epoch
        src_acc = evaluate(source_loader, name="Source", print_report=False)
        tgt_acc = evaluate(target_loader, name="Target", print_report=False)

        if tgt_acc > best_acc:
            best_acc = tgt_acc
            torch.save(model.state_dict(), r"UDA_MRI\Model weights\best_dann_model.pth")

        # log line
        log_line = f"Epoch [{epoch+1}/{num_epochs}] " \
                   f"Loss: {total_loss/n_batches:.4f}, " \
                   f"Cls Loss: {total_cls/n_batches:.4f}, " \
                   f"Dom Loss: {total_dom/n_batches:.4f}, " \
                   f"Source Acc: {src_acc:.2f}%, " \
                   f"Target Acc: {tgt_acc:.2f}%\n"

        print(log_line.strip())
        log_file.write(log_line)
        log_file.flush()

        # save metrics for .pth
        metrics["loss"].append(total_loss / n_batches)
        metrics["cls_loss"].append(total_cls / n_batches)
        metrics["dom_loss"].append(total_dom / n_batches)
        metrics["src_acc"].append(src_acc)
        metrics["tgt_acc"].append(tgt_acc)

    # save all metrics as .pth file
    torch.save(metrics, r"UDA_MRI\Outputs\DANN_log.pth")


# ---------------------------
# Run training and evaluation
# ---------------------------
train(num_epochs=100)
evaluate(target_loader, name="Target", print_report=True)
log_file.close()
