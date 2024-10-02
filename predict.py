import torch
from torchvision import transforms, models
from PIL import Image
import json
import cv2
import numpy as np

# Define the model architecture (VGG16)
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 8)

# Load the state dictionary
state_dict = torch.load('model_10.pth', map_location=torch.device('cpu'))
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


def predict_race(frame):
    # Check if the frame is valid
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("Invalid frame: None or not a numpy array")
    # Convert the OpenCV BGR frame to RGB and then to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Decode the prediction
    race = race_labels[str(predicted.item())]
    return race

def predict_race_test(imgpath):
    # Convert the OpenCV BGR frame to RGB and then to a PIL Image
    image = Image.open(imgpath)

    # Preprocess the image
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Decode the prediction
    race = race_labels[str(predicted.item())]
    return race

# # Example usage
# image_path = 'Untitled.png'
# race = predict_race_test(image_path)
# print(f'The predicted race is: {race}')