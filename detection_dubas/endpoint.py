from django.shortcuts import render
from django.conf import settings
from ultralytics import YOLO
import os
from PIL import Image
import shutil
import glob

# Chemin absolu vers le modèle
MODEL_PATH = "NO_MODEL_PROVIDED"
model = YOLO(MODEL_PATH)

def detect_dubas(request):
    prediction_image_url = None

    if request.method == 'POST' and 'image' in request.FILES:
        image = request.FILES['image']

        # Créer le chemin absolu vers le dossier static
        static_dir = os.path.join(settings.BASE_DIR, 'dubasdetection', 'static')
        os.makedirs(static_dir, exist_ok=True)

        # Chemin complet de l’image temporaire
        input_path = os.path.join(static_dir, 'input.jpg')

        # Sauvegarder l’image uploadée
        with open(input_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # Utiliser YOLO pour détecter
        results = model.predict(input_path, save=True, save_txt=False)

        # Trouver la dernière image générée par YOLO
        pred_images = glob.glob('runs/detect/*/input.jpg')
        if pred_images:
            last_pred_path = max(pred_images, key=os.path.getctime)
            output_path = os.path.join(static_dir, 'output.jpg')
            shutil.copy(last_pred_path, output_path)

            # URL pour afficher l’image
            prediction_image_url = os.path.join(settings.STATIC_URL, 'output.jpg')
        else:
            prediction_image_url = None  # Aucun résultat trouvé

    return render(request, 'template.html', {
        'prediction_image_url': prediction_image_url
    })
