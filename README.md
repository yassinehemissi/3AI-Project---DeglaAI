## DeglaAI

*This repository presents the version 0.1 of our work with only the Django endpoints (API Routes); the new frontend, models checkpoints and chatbot weren't pushed into this public repository*

### **Overview**
DeglaAI is a Django-based web platform developed as part of the 3AI Project at ESPRIT University. It represents the conclusion of an AI-centered initiative focused on revolutionizing date palm agriculture using modern machine learning techniques. The system provides tools for pest detection, quality classification, disease diagnosis, infestation prediction, and even a chatbot to assist farmers in real-time.

### **Features**
- Date Palm Quality Classification
- Pest Detection
- Leaf Disease Detection
- Infestation Prediction
- Chatbot for Agricultural Support

### **Tech Stack**
#### **Frontend**
- Django Templates (HTML, CSS (Bootstrap), JavaScript)
- Next.js (Tailwind + ShadCN)

#### **Backend**
- Django (Python)
- Django Rest Framework (if used for APIs)


### Directory Structure
```
3AI-Project---DeglaAI/
  classification_date_quality/
    endpoint.py
    models.py
  classification_date_variety/
    endpoint.py
    predict.py
  classification_leaves_diseases/
    endpoint.py
  classification_white_scale/
    endpoint.py
  detection_boufaraoui/
    endpoint.py
  detection_dubas/
    endpoint.py
  detection_pest/
    endpoint.py
  README.md
```
### Getting Started
1. Create a Django Project
```bash
python -m django --version
django-admin startproject main deglaai 
```
2. Create an app 
```bash
python manage.py startapp app_name
```
3. Copy one of the endpoint code (with other python files if they exit and make `endpoint.py` code be an import into your views.py as a `View`)

*Note the models aren't provided with this repo but you can still train your own models based on the architecture provided in the endpoints* 
### **Acknowledgments**

This project was developed by students (SynapTech Team) at ESPRIT University as part of the 3AI Project initiative. Special thanks to the faculty and our mentors miss **Sonia Mesbeh** and miss **Jihene Hlel** who guided the research and development phases.