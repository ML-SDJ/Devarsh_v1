from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Lambda
import os

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),         # Resize to 128x128
    Lambda(lambda img: img.convert("RGB")),  # Convert grayscale to 3 channels
    transforms.ToTensor(),                 # Convert to tensor (automatically normalizes to [0,1])
    transforms.Normalize([0.5], [0.5])     # Shift to [-1,1] range; good for CNNs
])


train_dataset = ImageFolder(root='data/chest_xray/train', transform=data_transforms)
val_dataset = ImageFolder(root='data/chest_xray/val', transform=data_transforms)
test_dataset = ImageFolder(root='data/chest_xray/test', transform=data_transforms)

img, label = train_dataset[0]  # Get first image and label
# If image is grayscale, squeeze to remove single channel
plt.imshow(img.permute(1, 2, 0).numpy()) # No cmap for RGB
plt.title(f'Label: {label} (0=NORMAL, 1=PNEUMONIA)')
plt.show()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32) #to flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

trian_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in trian_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

model.eval()
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)  # batch of predictions
        predictions.extend(preds.cpu().numpy())


model = SimpleCNN()
model.load_state_dict(torch.load('cnn_chestxray.pth', map_location=device))
model.to(device)
model.eval()


folder = 'data/chest_xray/test/NORMAL/'
model.eval()
results = []

for filename in os.listdir(folder):
    if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        input_tensor = data_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()  # 0=NORMAL, 1=PNEUMONIA
        results.append((filename, pred))
        print(f'{filename}: {pred}')
