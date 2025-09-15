import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# ---------------------------
# Feature Extractor + Classifier
# ---------------------------
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
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
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(DomainDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 domains: source/target
        )

    def forward(self, x):
        return self.fc(x)


# ---------------------------
# CDAN loss
# ---------------------------
def conditional_adversarial_loss(features, class_preds, domain_discriminator):
    softmax_output = nn.Softmax(dim=1)(class_preds)
    op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
    op_out = op_out.view(op_out.size(0), -1)
    domain_logits = domain_discriminator(op_out)
    return domain_logits


# ---------------------------
# Data Loading
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

source_dataset = datasets.ImageFolder(root=r"UDA_MRI\Datasets\Source", transform=transform)
target_dataset = datasets.ImageFolder(root=r"UDA_MRI\Datasets\Target", transform=transform)

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)


# ---------------------------
# Model Setup
# ---------------------------
feature_extractor = FeatureExtractor().cuda()
classifier = Classifier().cuda()
domain_discriminator = DomainDiscriminator(input_dim=64*53*53*3).cuda()

criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(feature_extractor.parameters()) +
    list(classifier.parameters()) +
    list(domain_discriminator.parameters()), lr=1e-4
)

best_acc = 0.0
log_history = {"epochs": []}  # <-- dict for saving logs


# ---------------------------
# Evaluation (Generic)
# ---------------------------
def evaluate(loader, dataset, print_report=True):
    feature_extractor.eval()
    classifier.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.cuda(), labels.cuda()
            features = feature_extractor(data)
            preds = classifier(features.view(features.size(0), -1))
            pred_labels = preds.argmax(1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    report = classification_report(all_labels, all_preds, target_names=dataset.classes, output_dict=True)

    if print_report:
        print(f"Accuracy: {acc:.2f}%")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=dataset.classes))

    return acc, report


# ---------------------------
# Training
# ---------------------------
def train(num_epochs=20):
    global best_acc, log_history
    for epoch in range(num_epochs):
        feature_extractor.train()
        classifier.train()
        domain_discriminator.train()

        total_cls_loss, total_dom_loss = 0, 0

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        n_batches = min(len(source_iter), len(target_iter))

        for i in range(n_batches):
            source_data, source_labels = next(source_iter)
            target_data, _ = next(target_iter)

            source_data, source_labels = source_data.cuda(), source_labels.cuda()
            target_data = target_data.cuda()

            # Source forward
            src_features = feature_extractor(source_data)
            src_features_flat = src_features.view(src_features.size(0), -1)
            src_class_preds = classifier(src_features_flat)
            cls_loss = criterion_class(src_class_preds, source_labels)

            # Target forward
            tgt_features = feature_extractor(target_data)
            tgt_features_flat = tgt_features.view(tgt_features.size(0), -1)
            tgt_class_preds = classifier(tgt_features_flat)

            # Domain loss (source)
            src_domain_logits = conditional_adversarial_loss(src_features_flat, src_class_preds, domain_discriminator)
            dom_loss_src = criterion_domain(src_domain_logits, torch.zeros(len(source_data), dtype=torch.long).cuda())

            # Domain loss (target)
            tgt_domain_logits = conditional_adversarial_loss(tgt_features_flat, tgt_class_preds, domain_discriminator)
            dom_loss_tgt = criterion_domain(tgt_domain_logits, torch.ones(len(target_data), dtype=torch.long).cuda())

            dom_loss = dom_loss_src + dom_loss_tgt
            loss = cls_loss + dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_dom_loss += dom_loss.item()

        # Evaluate after each epoch
        src_acc, src_report = evaluate(source_loader, source_dataset, print_report=False)
        tgt_acc, tgt_report = evaluate(target_loader, target_dataset, print_report=False)

        if tgt_acc > best_acc:
            best_acc = tgt_acc
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'domain_discriminator': domain_discriminator.state_dict(),
            }, r"UDA_MRI\Model weights\cdan_model_weights.pth")

        # Store logs
        log_history["epochs"].append({
            "epoch": epoch + 1,
            "cls_loss": total_cls_loss / n_batches,
            "dom_loss": total_dom_loss / n_batches,
            "source_acc": src_acc,
            "target_acc": tgt_acc,
            "source_report": src_report,
            "target_report": tgt_report
        })

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Cls Loss: {total_cls_loss/n_batches:.4f}, "
              f"Dom Loss: {total_dom_loss/n_batches:.4f}, "
              f"Source Acc: {src_acc:.2f}%, Target Acc: {tgt_acc:.2f}%")

        # Save logs after every epoch
        torch.save(log_history, r"UDA_MRI\Outputs\cdan_log.pth")

    print("Training finished. Logs saved in 'cdan_log.pth'")


# ---------------------------
# Run Training & Evaluation
# ---------------------------
train(num_epochs=100)

print("\nFinal Source Evaluation:")
evaluate(source_loader, source_dataset, print_report=True)

print("\nFinal Target Evaluation:")
evaluate(target_loader, target_dataset, print_report=True)
