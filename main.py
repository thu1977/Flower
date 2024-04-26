import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import densenet169, DenseNet169_Weights
from torch.utils.data import Dataset, DataLoader
from skimage.feature import hog
from skimage import color
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Preprocessing of the images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class Flowers102Dataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.dataset = torchvision.datasets.Flowers102(root=root, split=split, download=True, transform=None)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        image_np = np.array(image.permute(1, 2, 0))
        image_hsv = color.rgb2hsv(image_np)
        fd, hog_image = hog(image_hsv, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            channel_axis=-1, visualize=True)
        # print("HOG feature dimension:", fd.shape)
        fd = torch.tensor(fd, dtype=torch.float32)
        return image, fd, target

class MyDenseNet(nn.Module):
    def __init__(self, num_classes=102, num_hog_features=1568):
        super(MyDenseNet, self).__init__()
        # Load the DenseNet model
        self.densenet = densenet169(weights=DenseNet169_Weights.DEFAULT)
        num_ftrs = self.densenet.classifier.in_features
        # Add a new classifier layer, with the number of input features equal to
        # the sum of DenseNet and HOG features
        self.classifier = nn.Linear(num_ftrs + num_hog_features, num_classes)

    def forward(self, x, hog_features):
        features = self.densenet.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # Combine the DenseNet and HOG features
        combined_features = torch.cat((out, hog_features), dim=1)
        out = self.classifier(combined_features)
        return out

# Load the training dataset (processed)
train_dataset = Flowers102Dataset(root='./data', split='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = MyDenseNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# eta scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Validation
val_dataset = Flowers102Dataset(root='./data', split='val', transform=test_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # Operations inside don't track history
        for inputs, hog_features, labels in dataloader:
            inputs = inputs.to(device)
            hog_features = hog_features.to(device)
            labels = labels.to(device)

            outputs = model(inputs, hog_features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    return total_loss, total_acc


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=25):
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        train_loss = 0.0
        train_corrects = 0

        # Iterate over data.
        for inputs, hog_features, labels in train_loader:
            inputs = inputs.to(device)
            hog_features = hog_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs, hog_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        print(
            f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # Deep copy the model if it has the best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_wts)
    total_time = time.time() - start_time
    print(f'Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    return model

if __name__ == "__main__":
    model_ft = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=20)
    torch.save(model_ft.state_dict(), 'best_model_flowers.pth')
