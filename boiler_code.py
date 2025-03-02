import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, random_split

# 1. Model Definition and Adaptation for CIFAR-10
def get_cifar10_resnet18():
    # Load a pretrained ResNet-18 model.
    model = resnet18(pretrained=True)
    
    # Adapt the first conv layer: use 3x3 kernel, stride=1, and padding=1 
    # (CIFAR-10 images are only 32x32, so we don't want too aggressive downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Replace the maxpool with an identity operation (to preserve spatial resolution)
    model.maxpool = nn.Identity()
    
    # Modify the fully connected layer to output 10 classes instead of 1000
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_cifar10_resnet18().to(device)

# 2. Freeze All Layers Except conv1 and fc
for param in model.parameters():
    param.requires_grad = False
for param in model.conv1.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# 3. Data Preparation for CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download CIFAR-10 dataset (train split will be further split into training and validation)
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Split training dataset into training and validation (e.g., 90% train, 10% val)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Define Loss and Optimizer (only parameters with requires_grad=True are updated)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 5. Training Step: One Epoch of Training
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # ensure model is in training mode
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)       # raw logits output
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        total_correct += torch.sum(preds == labels).item()
        total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 6. Validation/Testing Step: Evaluate Without Backpropagation
def evaluate(model, dataloader, criterion, device):
    model.eval()  # set to evaluation mode
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # raw logits output
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# 7. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}]:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_acc:.4f}")

# 8. Testing After Training
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
