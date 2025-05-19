# Step 4: Create a view to handle the image upload and prediction
# In your Django app directory (e.g., date_classifier/views.py)

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import DateClassification, get_predictor
from .forms import DateImageUploadForm
from PIL import Image
import json
import base64
import io


def classify_fruit_quality(request):
    """
    View for the Classify Fruit Quality page
    """
    if request.method == 'GET':
        form = DateImageUploadForm()
        return render(request, 'DateQuality/classify_quality.html', {'form': form})
    
    elif request.method == 'POST':
        form = DateImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the form but don't commit to DB yet
            date_classification = form.save(commit=False)
            
            # Get the predictor
            predictor = get_predictor()
            
            # Open the image
            image = Image.open(date_classification.image)
            
            # Make prediction
            prediction = predictor.predict(image)
            
            # Update model with prediction results
            date_classification.predicted_class = prediction['class']
            date_classification.confidence = prediction['confidence']
            
            # Save to DB
            date_classification.save()
            
            # Return result to frontend
            context = {
                'form': DateImageUploadForm(),
                'result': {
                    'class': prediction['class'],
                    'confidence': f"{prediction['probability']:.2%}",
                    'all_probabilities': {k: f"{v:.2%}" for k, v in prediction['all_probabilities'].items()}
                },
                'image_url': date_classification.image.url
            }
            return render(request, 'DateQuality/classify_quality.html', context)
        
        return render(request, 'DateQuality/classify_quality.html', {'form': form})


@csrf_exempt
def api_classify_fruit_quality(request):
    """
    API endpoint for AJAX requests
    """
    if request.method == 'POST':
        try:
            # Check if request contains an image file
            if 'image' in request.FILES:
                image_file = request.FILES['image']
                image = Image.open(image_file)
            
            # Or if it contains a base64 image
            elif 'image_data' in request.POST:
                image_data = request.POST['image_data']
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                return JsonResponse({'error': 'No image provided'}, status=400)
            
            # Get prediction
            predictor = get_predictor()
            prediction = predictor.predict(image)
            
            # Format response
            response = {
                'success': True,
                'class': prediction['class'],
                'confidence': prediction['probability'],
                'all_probabilities': prediction['all_probabilities']
            }
            
            return JsonResponse(response)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)