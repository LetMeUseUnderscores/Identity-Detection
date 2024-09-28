import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class FairFaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.face_frame.iloc[:, 3])  # Assuming the racial group is in the fourth column

    def __len__(self):
        return len(self.face_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0].split('/')[-1])
        image = Image.open(img_name)
        racial_group = self.face_frame.iloc[idx, 3]  # Assuming the racial group is in the fourth column

        if self.transform:
            image = self.transform(image)

        racial_group_encoded = self.label_encoder.transform([racial_group])[0]
        racial_group_encoded = torch.tensor(racial_group_encoded, dtype=torch.long)  # Convert to torch.long

        return image, racial_group_encoded


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = FairFaceDataset(csv_file='Data\\FairFace\\train_labels.csv',
                          root_dir='Data\\FairFace\\train',
                          transform=transform)

val_dataset = FairFaceDataset(csv_file='Data\\FairFace\\val_labels.csv',
                              root_dir='Data\\FairFace\\val',
                              transform=transform)

# Load the pre-trained VGG-Face model
model = models.vgg16(pretrained=False)

# Number of classes in dataset
num_classes = 8

# Replace the classifier layer
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.load_state_dict(torch.load("model_1.pth"))

# Freeze early layers of the model, fine-tune the later layers and the classifier
for param in model.features.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
for param in model.classifier.parameters():
    param.requires_grad = True

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

# Optional: Define a learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)


# Code to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=9):
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Must use CUDA-equipped GPU for training or else code will not work!!!!!!

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 100 == 0:
                    print(f'Batch {i + 1}, Loss: {loss.item():.4f}')

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        torch.save(model.state_dict(), f'model_{epoch}.pth')
        print(f'Model saved to model_{epoch}.pth')
        print()

    print('Training complete')

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@%%%%%%%%%%%%@@@@@@@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%%@@@@@@@@@@@@@')
    print('@@@@@@@@@@@%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@')
    print('@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@')
    print('@@@@@@@@@%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%%@@@@@@@@@')
    print('@@@@@@@@%%###%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%#%%@@@@@@@@')
    print('@@@@@@@%#####%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%###%%@@@@@@@')
    print('@@@@@@@#####%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%#####%@@@@@@@')
    print('@@@@@@@#####%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%######@@@@@@@')
    print('@@@@@@@#####%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%#####@@@@@@@')
    print('@@@@@@@####%%%%%%%%%%%##****##%%%%%%%%%%%##****##%%%%%%%%%%#####@@@@@@@')
    print('@@@@@@@###%%#%%###*=:.       .-*#######+:.       .:=*######%%###@@@@@@@')
    print('@@@@@@@#*#%###**+:           ..:*#####*..           .-+*####%##@@@@@@@@')
    print('@@@@@@@@##%#***=.   ............*#####=...........    :+**#####@@@@@@@@')
    print('@@@@@@@@####***-................##%%%#=................+***###@@@@@@@@@')
    print('@@@@@@@@@######=:::::::::::....:%%%%%%*.....:::::::::::+#####%@@@@@@@@@')
    print('@@@@@@@@@%#####+:::---:::::::..+%%%%%%%-..::::::----::-#####%@@@@@@@@@@')
    print('@@@@@@@@@@%##%%#=-------:::::.+%%%%@%%%%-::::::-------*%%###@@@@@@@@@@@')
    print('@@@@@@@@@@@###%%#-----::::::+%%%%@@@@%%%%%-::::::----+%%%%#%@@@@@@@@@@@')
    print('@@@@@@@@@@@###%%%%#=-::-=+%@@%%%%@%#@@%%%%@@#+--::-+%@%%%%#%@@@@@@@@@@@')
    print('@@@@@@@@@@@######%%%%%%%%@@@%%%@*....:%%%%%%%%%%%%%%%%%%%##%@@@@@@@@@@@')
    print('@@@@@@@@@@@@***########%%%%%%%%=.::::..*%%%%%%%%%%#######*#@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@%*++++****##%%%%%#*==*#*=+#%%%%%####*****++#@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@%#+++*###%%%%%%%%%%%%%%%####***+*#%@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@#+***####%%%%%%%%%%%####***+@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@+****######%%%%%%%####*****@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@#++******##########***+++*@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@%*%%*#%#*******#***%%*#%##@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@#%%#%@@%%@@@#@@@%%@@%%%##@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@#%#%@@%@@@@#@@@%%@@###%@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@*#%%@@#%@@%%%@@%%@@%%##@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@%++#@@#@@@@%@@@%%@%*+*@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@#***##****#****##***%@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@#**###########**#@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%######%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return model


# Epoch 1:
# train Loss: 1.6515 Acc: 0.3667
# val Loss: 1.5006 Acc: 0.4375

# Epoch 2:
# train Loss: 1.5079 Acc: 0.4271
# val Loss: 1.5027 Acc: 0.4515

# Epoch 3:
# train Loss: 1.4346 Acc: 0.4583
# val Loss: 1.4892 Acc: 0.4505

# Epoch 4:
# train Loss: 1.3866 Acc: 0.4790
# val Loss: 1.5174 Acc: 0.4219

# Epoch 5:
# train Loss: 1.3312 Acc: 0.5018
# val Loss: 1.4958 Acc: 0.4234

# Epoch 6:
# train Loss: 1.2842 Acc: 0.5225
# val Loss: 1.4981 Acc: 0.4345

# Epoch 7:
# train Loss: 1.2376 Acc: 0.5403
# val Loss: 1.5063 Acc: 0.4261

# Epoch 8:
# train Loss: 1.1932 Acc: 0.5598
# val Loss: 1.5082 Acc: 0.4154

# Epoch 9:
# train Loss: 0.9036 Acc: 0.6615
# val Loss: 1.3673 Acc: 0.4859

# Epoch 10:
# train Loss: 0.7957 Acc: 0.7013
# val Loss: 1.3743 Acc: 0.4907

# Call the train_model function
model_ft = train_model(model, criterion, optimizer, scheduler, num_epochs=9)
