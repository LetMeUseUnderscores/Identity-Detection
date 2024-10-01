import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

# Assuming you have a dataset class 'YourDatasetClass'

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

# Step 1: Load the model
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 8)

state_dict = torch.load('model_10.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Step 2: Prepare the data
transform = transforms.Compose([
    # Add necessary transformations here
    transforms.ToTensor(),
])
val_dataset = FairFaceDataset(csv_file='Data\\FairFace\\val_labels.csv',
                              root_dir='Data\\FairFace\\val',
                              transform=transform)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Step 3: Make predictions
all_preds = []
all_labels = []
with torch.no_grad():
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Step 4: Calculate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Step 5: Visualize the confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
plt.show()