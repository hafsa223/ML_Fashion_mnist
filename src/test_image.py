import joblib
import numpy as np
from preprocess import load_and_preprocess_data
import matplotlib.pyplot as plt

def test_single_image(image_path):
    model = joblib.load('models/best_model.sav')
    # Charger une image personnalisée ici (par exemple, PIL ou OpenCV)
    # Prétraiter l'image pour la transformer en vecteur de 784 pixels
    image = ...  # Charger l'image ici
    image = image.reshape(1, -1) / 255.0
    prediction = model.predict(image)
    print(f"Prédiction : {prediction}")
