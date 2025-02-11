import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
DATA_DIR = "dataset"
MODEL_SAVE_PATH = "anomaly_detection_model.pth"

# Parameters
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001

# Data Augmentation and Normalization
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets
datasets = {
    x: datasets.ImageFolder(
        os.path.join(DATA_DIR, x),
        data_transforms[x]
    )
    for x in ["train", "val"]
}

dataloaders = {
    x: DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    for x in ["train", "val"]
}

# Model: Fine-Tune Pretrained ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(datasets["train"].classes))  # Adjust for anomaly classes
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders[phase]):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (torch.argmax(outputs, 1) == labels).sum().item()

        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects / len(datasets[phase])

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# Save Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved.")
