from django.shortcuts import render
from .forms import DateImageUploadForm
from .predict import predict_date_variety
import os
from django.conf import settings

def classify_date(request):
    prediction = None
    confidence = None
    if request.method == 'POST':
        print("Received POST request")  # Debug
        form = DateImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            print("Form is valid")  # Debug
            static_dir = os.path.join(settings.BASE_DIR, 'date_variety', 'static')
            os.makedirs(static_dir, exist_ok=True)

            img = form.cleaned_data['image']
            # Chemin complet de l’image temporaire
            img_path = os.path.join(static_dir, img.name)

            print(f"Saving image to: {img_path}")  # Debug
            # Save image
            try:
                with open(img_path, 'wb+') as f:
                    for chunk in img.chunks():
                        f.write(chunk)
                print(f"Image saved successfully")  # Debug
                # Predict
                prediction, confidence = predict_date_variety(img_path)
                print(f"Prediction: {prediction}")  # Debug
            except Exception as e:
                print(f"Error saving image or predicting: {e}")  # Debug
        else:
            print("Form is invalid:", form.errors)  # Debug
    else:
        print("Rendering form (GET request)")  # Debug
        form = DateImageUploadForm()
    return render(request, 'date_variety/classify.html', {
        'form': form,
        'prediction': prediction,
        'confidence': confidence,
        'img_path': '/static/' + img.name if 'img_path' in locals() else None,
        
    })