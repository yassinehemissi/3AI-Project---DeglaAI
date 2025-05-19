from django.shortcuts import render
from django.conf import settings
from ultralytics import YOLO
import os
from PIL import Image
import shutil
import glob
import uuid
import time
from django.http import StreamingHttpResponse, JsonResponse
import cv2
import threading
from ultralytics import YOLO

from django.http import JsonResponse
import json
from datetime import datetime, timedelta
from pygame import mixer

# Global variables for camera
camera_thread = None
camera_running = False

# Variable globale pour stocker les dernières détections
latest_detections = []
last_detection_update = datetime.now()

# Initialisation de l'audio
mixer.init()
alarm_sound = mixer.Sound(os.path.join(settings.BASE_DIR, 'boufaroua', 'static', 'audio', 'alarm.wav'))

# Variables globales
latest_detections = []
last_detection_update = datetime.now()
is_alarm_playing = False
last_detection_time = 0

# Chemin absolu vers le modèle
MODEL_PATH = "NO_MODEL_PROVIDED"
model = YOLO(MODEL_PATH)

def boufaroua(request):
    prediction_image_url = None
    detection_results = None

    if request.method == 'POST' and 'image' in request.FILES:
        image = request.FILES['image']
        
        # Créer un identifiant unique pour cette session
        session_id = str(uuid.uuid4())[:8]
        
        # Créer les dossiers nécessaires
        static_dir = os.path.join(settings.BASE_DIR, 'boufaroua', 'static', 'detections')
        os.makedirs(static_dir, exist_ok=True)
        
        # Dossier pour cette session spécifique
        session_dir = os.path.join(static_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Chemin pour l'image d'entrée
        input_path = os.path.join(session_dir, 'input.jpg')
        
        # Sauvegarder l'image uploadée
        with open(input_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        
        # Effectuer la prédiction
        results = model.predict(input_path, save=True, save_txt=True, project=session_dir, name='output')
        
        # Récupérer les résultats de détection
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        # Trouver l'image de sortie
        output_image_path = os.path.join(session_dir, 'output', 'input.jpg')
        if os.path.exists(output_image_path):
            # Chemin relatif pour le template
            relative_path = os.path.join('detections', session_id, 'output', 'input.jpg')
            prediction_image_url = os.path.join(settings.STATIC_URL, relative_path)
        
        # Préparer les résultats pour le template
        if detections:
            detection_results = {
                'count': len(detections),
                'items': detections,
                'has_boufaroua': any(d['class'] == 'boufaroua' for d in detections)
            }
        else:
            detection_results = {
                'count': 0,
                'items': [],
                'has_boufaroua': False
            }

    return render(request, 'boufaroua_detection.html', {
        'prediction_image_url': prediction_image_url,
        'detection_results': detection_results
    })
    
def gen_frames():
    cap = cv2.VideoCapture(0)
    global latest_detections, last_detection_update, is_alarm_playing, last_detection_time
    
    while camera_running:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform detection
        results = model.predict(frame, conf=0.5)
        
         # Process detections
        detections = []
        has_target = False
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': bbox
                })
                
                # Vérifie si c'est boufaroua ou oeuf
                if class_name.lower() in ['boufaroua', 'oeuf']:
                    has_target = True
                
                # Draw bounding box
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                # Choose color based on class (red for boufaroua/oeuf, green for others)
                # color = (0, 0, 255) if class_name.lower() in ['boufaroua', 'oeuf'] else (0, 255, 0)
                # Assign unique color for each class
                if class_name.lower() == 'boufaroua':
                 color = (0, 0, 255)  # Rouge pour boufaroua
                elif class_name.lower() == 'oeuf':
                 color = (0, 255, 0)  # Vert pour oeuf
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1 = max(y1, label_size[1])
                
                # Draw filled rectangle for text background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                
                # Draw text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mettre à jour les dernières détections
        latest_detections = detections
        last_detection_update = datetime.now()
        
        # Gestion de l'alarme
        if has_target:
            if not is_alarm_playing:
                alarm_sound.play(-1)  # Jouer en boucle
                is_alarm_playing = True
            last_detection_time = time.time()
            cv2.putText(frame, "ALERTE!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if is_alarm_playing and (time.time() - last_detection_time > 2.0):
                alarm_sound.stop()
                is_alarm_playing = False
        
        # Encoder le frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

def camera_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def start_camera(request):
    global camera_running, camera_thread
    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(target=gen_frames)
        camera_thread.daemon = True
        camera_thread.start()
        return JsonResponse({'status': 'started'})
    return JsonResponse({'status': 'already running'})

def stop_camera(request):
    global camera_running
    camera_running = False
    return JsonResponse({'status': 'stopped'})

def predict_with_camera(request):
    global latest_detections, last_detection_update
    
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        # Effectuer la prédiction avec YOLO
        results = model.predict(image, conf=0.5)
        
        # Traiter les résultats
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id,
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                })
        
        # Mettre à jour les dernières détections
        latest_detections = detections
        last_detection_update = datetime.now()
        
        return JsonResponse({'detections': detections})
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_latest_detections(request):
    global latest_detections, last_detection_update
    
    # Supprimer les détections trop anciennes (plus de 5 secondes)
    if datetime.now() - last_detection_update > timedelta(seconds=5):
        latest_detections = []
    
    return JsonResponse({
        'detections': latest_detections,
        'last_update': last_detection_update.isoformat()
    })