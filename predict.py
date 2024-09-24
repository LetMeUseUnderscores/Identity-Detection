import torch
from torchvision import transforms, models
from PIL import Image
import json

# Define the model architecture (VGG16)
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 8)

# Load the state dictionary
state_dict = torch.load('model_1.pth')
model.load_state_dict(state_dict)
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the labels
with open('race_labels.json', 'r') as f:
    race_labels = json.load(f)

def predict_race(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Decode the prediction
    race = race_labels[str(predicted.item())]
    return race

# Example usage
image_path = 'path/to/image.jpg'
race = predict_race(image_path)
print(f'The predicted race is: {race}')