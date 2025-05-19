# Step 1: Create a model in Django to handle the predictions
# In your Django app directory (e.g., date_classifier/models.py)

from django.db import models
from django.conf import settings
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import io
import base64
from django.db import models  # Pour les modèles Django
import torchvision.models as torch_models  # Pour les modèles PyTorch

class DateClassification(models.Model):
    image = models.ImageField(upload_to='date_images/')
    predicted_class = models.CharField(max_length=100, blank=True)
    confidence = models.FloatField(default=0.0)
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2f})"


# Step 2: Create the model architecture that matches your trained model
class DateClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DateClassifier, self).__init__()
        # Load EfficientNet pre-trained
        self.efficientnet = torch_models.efficientnet_b0(pretrained=False)

        # Replace the last fully connected layer
        in_features = self.efficientnet.classifier[1].in_features

        # Create a new classifier with dropout
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


# Step 3: Create a predictor class to load the model and make predictions
class DateQualityPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the model
        self.load_model()
        
    def load_model(self):
        # Define the path to your model
        model_path = "NO_MODEL_PROVIDED"
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get class to index mapping and create index to class mapping
        class_to_idx = checkpoint['class_to_idx']
        self.class_names = {v: k for k, v in class_to_idx.items()}
        
        # Initialize the model
        self.model = DateClassifier(num_classes=len(class_to_idx))
        
        # Load the state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict(self, image_data):
        """
        Predict the class of an image.
        
        Args:
            image_data: PIL Image or path to image
            
        Returns:
            Dictionary with prediction results
        """
        # If image_data is a path, open the image
        if isinstance(image_data, str):
            image = Image.open(image_data).convert('RGB')
        else:
            image = image_data.convert('RGB')
            
        # Apply transformations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top prediction
            prob, predicted_class = torch.max(probabilities, 1)
            
            # Get all probabilities
            all_probs = {self.class_names[i]: probabilities[0][i].item() for i in range(len(self.class_names))}
            
        return {
            'class': self.class_names[predicted_class.item()],
            'probability': prob.item(),
            'all_probabilities': all_probs
        }


# Initialize the predictor (you'll use it as a singleton)
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = DateQualityPredictor()
    return predictor