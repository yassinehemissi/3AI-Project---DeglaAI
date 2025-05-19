from django.shortcuts import render
from django.conf import settings
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# Chemin absolu vers le modèle EfficientNet (PyTorch)
MODEL_PATH = "NO_MODEL_PROVIDED"

# Loading the model EfficientNet (PyTorch)
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)  # 2 classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model 

model = load_model()

# Classes Names Ranked
CLASS_NAMES = ["Brown Spots", 'Healthy', 'White Scale1', 'White Scale2', 'White Scale3']

def preprocess_image(image_path, target_size=(224, 224)):
	preprocess = transforms.Compose([
		transforms.Resize(target_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	img = Image.open(image_path).convert('RGB')
	img_tensor = preprocess(img).unsqueeze(0)
	return img_tensor

def detect_white_scale(request):
	prediction_label = None

	if request.method == 'POST' and 'image' in request.FILES:
		image = request.FILES['image']

		# Créer le chemin absolu vers le dossier static
		static_dir = os.path.join(settings.BASE_DIR, 'white_scale', 'static')
		os.makedirs(static_dir, exist_ok=True)

		# Chemin complet de l’image temporaire
		input_path = os.path.join(static_dir, 'input.jpg')

		# Sauvegarder l’image uploadée
		with open(input_path, 'wb+') as destination:
			for chunk in image.chunks():
				destination.write(chunk)

		# Prétraiter l’image et prédire avec EfficientNet (PyTorch)
		img_tensor = preprocess_image(input_path)
		with torch.no_grad():
			outputs = model(img_tensor)
			_, pred_class = torch.max(outputs, 1)
			prediction_label = CLASS_NAMES[pred_class.item()]

	return render(request, 'template.html', {
		'prediction_label': prediction_label
	})
