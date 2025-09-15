import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# -------------------------------
# Transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB case
])

# -------------------------------
# Dataset & Dataloader
# -------------------------------
trainset = torchvision.datasets.ImageFolder(
    root=r"UDA_MRI\Slice_Separation_Training_Testing\Training_binary",
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

#  create a separate test folder with unseen images
testset = torchvision.datasets.ImageFolder(
    root=r"UDA_MRI\Slice_Separation_Training_Testing\Testing_Source_binary",
    transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes = trainset.classes
print("Classes:", classes)


# -------------------------------
# Dilated CNN Model
# -------------------------------
class DilatedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DilatedCNN, self).__init__()

        # ---- Local Feature Extraction ----
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # normal conv
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # normal conv
        self.bn2 = nn.BatchNorm2d(64)

        # ---- Dilated Convolutions for Global Context ----
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2)  # dilated conv
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4)  # more dilation
        self.bn4 = nn.BatchNorm2d(256)

        # ---- Another convolution for deep features ----
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # ---- Global Average Pooling + Fully Connected ----
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Local features
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Dilated (global context)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        # Deep features
        x = F.leaky_relu(self.bn5(self.conv5(x)))

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -------------------------------
# Training Loop
# -------------------------------
def train_model(net, trainloader, criterion, optimizer, epochs=30):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), "UDA_MRI\Model weights\Slice_seperation.pth")
    print("Model saved as Slice_seperation.pth")


# -------------------------------
# Testing Loop
# -------------------------------
def test_model(net, testloader, classes):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on {total} test images: {100 * correct / total:.2f}%")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    net = DilatedCNN(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_model(net, trainloader, criterion, optimizer, epochs=50)
    test_model(net, testloader, classes)
