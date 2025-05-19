import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the model once
MODEL_PATH = "NO_MODEL_PROVIDED"
print(f"Loading model from: {MODEL_PATH}")  # Debug

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")  # Debug
except Exception as e:
    print(f"Error loading model: {e}")  # Debug

# Class names
CLASS_NAMES = ['Ajwa', 'allig', 'deglet_nour', 'Galaxy', 'Medjool', 'Meneifi', 'Nabtat Ali', 'Rutab', 'Shaishe', 'Sokari', 'Sugaey']

def predict_date_variety(img_path):
    print(f"Processing image: {img_path}")  # Debug
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        print(f"Image shape: {img_array.shape}")  # Debug
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        print(f"Input shape to model: {img_array.shape}")  # Debug
        predictions = model.predict(img_array)
        print(f"Predictions: {predictions}")  # Debug
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index]) * 100  # Convert to percentage
        print(f"Predicted class: {CLASS_NAMES[predicted_index]}, Confidence: {confidence}%")  # Debug
        return CLASS_NAMES[predicted_index], confidence
    except Exception as e:
        print(f"Error in prediction: {e}")  # Debug
        return None, None