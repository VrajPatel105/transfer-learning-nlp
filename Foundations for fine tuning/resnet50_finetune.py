from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
import torch.nn as nn
import torch

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # for ResNet 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225])   # ImageNet std
])

train_dataset = datasets.Flowers102(root='./data', split='train', transform=transform, download=True)
val_dataset = datasets.Flowers102(root='./data', split='val', transform=transform, download=True)


# print(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# print(f"Train size: {len(train_dataset)}")
# print(f"Val size: {len(val_dataset)}")

# Now build Version A — the fine-tuned model. Here's what you need to do:

# 1. Load ResNet-50 with pretrained weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
# 2. Freeze all layers except the last few
# we will freeze the first 4 layers and only the last two layers will be unfreezed since we have a very small dataset.
layer_groups = [model.conv1, model.layer1, model.layer2, model.layer3, model.layer4, model.fc]
freeze_layers = [model.conv1, model.layer1, model.layer2, model.layer3]

for layer in freeze_layers:
    for parameter in layer.parameters():
        parameter.requires_grad = False

# 3. Replace `model.fc` with a new linear layer that outputs 102 classes instead of 1000
# print(model.fc.out_features) -> 1000
model.fc = nn.Linear(model.fc.in_features, 102)
# print(model.fc.out_features) # -> 102

# print(model) -> confirmed that it has 102 output features
# lets also print the parameter count to confirm : 
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parametes : {total_params}")
# print(f"Trainable parametes : {trainable_params}")

# output : 
# Total parametes : 25557032
# Trainable parametes : 17013864

# 4. Use the layer-wise optimizer you built earlier
# since the first few layers are frozen, they will not have any effect.
base_lr = 1e-5
decay_factor = 3
param_list = []
for layer in layer_groups:
    param_list.append({
        'params': layer.parameters(),
        'lr' : base_lr
    })
    base_lr = base_lr*decay_factor

optimizer = Adam(params=param_list)

epochs = 30

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)

# training loop:
print("starting training looppp")
for epoch in range(epochs):

    total_loss = 0

    for feature, label in train_loader:

        feature, label = feature.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(feature)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        total_loss = total_loss + loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:4f}")



# val loop

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for feature, label in val_loader:
        feature, label = feature.to(device), label.to(device)
        output = model(feature)
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
val_acc = correct / total
print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")


# Outputt

# Epoch 1, Loss: 126.396499
# Epoch 2, Loss: 53.533345
# Epoch 3, Loss: 13.740381
# Epoch 4, Loss: 3.528853
# Epoch 5, Loss: 1.918117
# Epoch 6, Loss: 1.266922
# Epoch 7, Loss: 0.890517
# Epoch 8, Loss: 0.621674
# Epoch 9, Loss: 0.586107
# Epoch 10, Loss: 0.373697
# Epoch 11, Loss: 0.749382
# Epoch 12, Loss: 1.009658
# Epoch 13, Loss: 0.786297
# Epoch 14, Loss: 1.390330
# Epoch 15, Loss: 2.482102
# Epoch 16, Loss: 1.877763
# Epoch 17, Loss: 1.146052
# Epoch 18, Loss: 1.089588
# Epoch 19, Loss: 0.873604
# Epoch 20, Loss: 1.673916
# Epoch 21, Loss: 1.367489
# Epoch 22, Loss: 1.527590
# Epoch 23, Loss: 2.505540
# Epoch 24, Loss: 1.224399
# Epoch 25, Loss: 0.986638
# Epoch 26, Loss: 1.219706
# Epoch 27, Loss: 0.270791
# Epoch 28, Loss: 0.189315
# Epoch 29, Loss: 0.375046
# Epoch 30, Loss: 0.422005
# Epoch 30 | Train Loss: 0.4220 | Val Acc: 0.9137



# Version B : No freezing, no layer-wise LRs needed — just a flat learning rate since nothing is pretrained

model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102) 
learning_rate = 1e-3
optimizer = Adam(params=model.parameters(), lr=learning_rate)

epochs = 30

criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)

# training loop:
print("starting training looppp")
for epoch in range(epochs):

    total_loss = 0

    for feature, label in train_loader:

        feature, label = feature.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(feature)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        total_loss = total_loss + loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:4f}")



# val loop

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for feature, label in val_loader:
        feature, label = feature.to(device), label.to(device)
        output = model(feature)
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
val_acc = correct / total
print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

# output for version B :

# Epoch 1, Loss: 172.387294
# Epoch 2, Loss: 150.514360
# Epoch 3, Loss: 136.312006
# Epoch 4, Loss: 127.990803
# Epoch 5, Loss: 122.688704
# Epoch 6, Loss: 116.721842
# Epoch 7, Loss: 114.454754
# Epoch 8, Loss: 107.227245
# Epoch 9, Loss: 100.680824
# Epoch 10, Loss: 98.673928
# Epoch 11, Loss: 95.122265
# Epoch 12, Loss: 88.203417
# Epoch 13, Loss: 88.697072
# Epoch 14, Loss: 87.264764
# Epoch 15, Loss: 84.140084
# Epoch 16, Loss: 77.449211
# Epoch 17, Loss: 76.697827
# Epoch 18, Loss: 72.502982
# Epoch 19, Loss: 73.216885
# Epoch 20, Loss: 69.341692
# Epoch 21, Loss: 65.362300
# Epoch 22, Loss: 62.538006
# Epoch 23, Loss: 61.795145
# Epoch 24, Loss: 56.934735
# Epoch 25, Loss: 53.967737
# Epoch 26, Loss: 53.372801
# Epoch 27, Loss: 51.050700
# Epoch 28, Loss: 45.797018
# Epoch 29, Loss: 46.424568
# Epoch 30, Loss: 41.510976
# Epoch 30 | Train Loss: 41.5110 | Val Acc: 0.3069