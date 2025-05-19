from django.shortcuts import render
import joblib
import pandas as pd
import serial

# Charger le modèle
MODEL_PATH = "NO_MODEL_PROVIDED"
pipeline = joblib.load(MODEL_PATH)

RAVAGEUR_LABELS = {
'6': "Date Palm Borer",
'1': "Boufarwa",
'2': "Red Palm Weevil",
'3': "Desert Locust",
'4': "Date Palm Mite",
'5': "Scale Insects",
'0': "None"

}

insect_data = {
    '6': {
        'description': "Papillon nocturne dont la larve attaque les dattes, provoquant de gros dégâts dans les palmeraies.",
        'solutions': "Utiliser des pièges à phéromones, appliquer des traitements biologiques.",
        'image_url': '/static/images/pyrale.jpg'
    },
    '1': {
        'description': "Ravageur des céréales (surtout blé et orge) qui dévore les jeunes pousses.",
        'solutions': "Lutter par rotation des cultures, pulvérisations spécifiques si infestation.",
        'image_url': '/static/images/boufarwa.jpg'
    },
    '2': {
        'description': "Insecte destructeur de palmiers, creusant des galeries internes qui tuent l’arbre.",
        'solutions': "Surveillance régulière, piégeage, abattage et brûlage des palmiers infectés.",
        'image_url': '/static/images/charancon.jpg'
    },
    '3': {
        'description': "Insecte migrateur dévastateur qui détruit les cultures sur des kilomètres carrés.",
        'solutions': "Intervention rapide par traitements aériens, surveillance satellite.",
        'image_url': '/static/images/criquet.jpg'
    },
    '4': {
        'description': "Petit acarien qui attaque les fruits du palmier, entraînant des pertes de qualité.",
        'solutions': "Traitement acaricide homologué, techniques de lutte intégrée.",
        'image_url': '/static/images/mite.jpg'
    },
    '5': {
        'description': "Groupe d'insectes suceurs s’attaquant à la sève des plantes, affaiblissant les arbres.",
        'solutions': "Application d'huiles horticoles, contrôle biologique par insectes prédateurs.",
        'image_url': 'static/img/phoenicococcus.jpg'
    },
    '0': {
        'description': "Aucun insecte détecté dans l'échantillon soumis.",
        'solutions': "-",
        'image_url': '../static/images/aucun.png'
    }
}





def lire_donnees_capteur(port='COM3', baudrate=9600):
    try:
        with serial.Serial(port, baudrate, timeout=3) as arduino:
            line = arduino.readline().decode('utf-8').strip()
            print("Données capteur brutes :", line)

            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Format de données inattendu : {line}")
            
            temperature = float(parts[0])
            humidite = float(parts[1])
            vitesse_vent = float(parts[2])
            return temperature, humidite, vitesse_vent

    except Exception as e:
        print("Sensor reading error:", e)
        return None, None, None


def predict(request):
    prediction_label = None
    insect_info = None
    error_message = None

    if request.method == 'POST':
        try:
            temperature, humidite, vitesse_vent = lire_donnees_capteur()
            if None in (temperature, humidite, vitesse_vent):
                error_message = "Sensor reading error."
            else:
                data = pd.DataFrame([[temperature, humidite, vitesse_vent]],
                                    columns=['Temperature_C', 'Humidite_percent', 'Vitesse_vent_kmh'])

                prediction_array = pipeline.predict(data)
                prediction_index = str(prediction_array[0])
                prediction_label = RAVAGEUR_LABELS.get(prediction_index, "Ravageur inconnu")
                insect_info = insect_data.get(prediction_index)

        except Exception as e:
            error_message = f"Erreur lors de la prédiction : {str(e)}"

    return render(request, 'predict.html', {
        'prediction_label': prediction_label,
        'insect_info': insect_info,
        'error_message': error_message
    })



def insectes_ecailles(request):
    return render(request, 'insectes-ecailles.html')

def boufarwa(request):
    return render(request, 'boufarwa.html')