import os
import numpy as np
from django.conf import settings
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input


# Load the model once globally
MODEL_PATH =  "NO_MODEL_PROVIDED"
model = load_model(MODEL_PATH, custom_objects={'preprocess_input': preprocess_input})

# Define class names (update with your actual class labels)
CLASS_NAMES = ['1. Potassium Deficiency', '2. Manganese Deficiency', '3. Magnesium Deficiency', '4. Black Scorch', '5. Leaf Spots', '6. Fusarium Wilt', '7. Rachis Blight', '8. Parlatoria Blanchardi', '9. Healthy sample']
DISEASE_DESCRIPTIONS = {
    'Black Scorch': 'A fungal disease causing dark lesions and leaf decay, common in warm climates.',
    'Fusarium Wilt': 'Caused by Fusarium fungus, leading to yellowing and wilting of leaves.',
    'Healthy': 'No signs of disease detected. The palm appears healthy and vibrant.',
    'Leaf Spots': 'Small brown or black spots on leaves, often fungal or bacterial in origin.',
    'Magnesium Deficiency': 'Yellowing of older leaves, especially between veins.',
    'Manganese Deficiency': 'Frizzle top appearance with distorted new leaves.',
    'Parlatoria Blanchardi': 'A pest that causes yellow patches and weakens the palm over time.',
    'Potassium Deficiency': 'Margins of older leaves turn yellow or necrotic; affects fruit quality.',
    'Rachis Blight': 'Necrosis or discoloration of the rachis, weakening the frond structure.'
}




def predictt(request):
    prediction = None
    description = None
    image_url = None
    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']
        img_path = os.path.join(settings.MEDIA_ROOT, 'tempe.png')

        # Save uploaded image
        with open(img_path, 'wb+') as f:
            for chunk in img_file.chunks():
                f.write(chunk)

        # Load and preprocess image
        # Image preprocessing
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # 💡 USE THIS instead of /255.0

        # Make prediction
        preds = model.predict(img_array)
        prediction = CLASS_NAMES[np.argmax(preds)]

    return render(request, 'predict2.html', {
    'prediction': prediction,
    'description': description,
    'image_url': image_url,
    'disease_info': DISEASE_DESCRIPTIONS  # This will contain ALL diseases
})
