from preprocess import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Charger les données
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Charger le modèle
model = joblib.load('models/best_model.sav')

# Évaluation
y_pred = model.predict(X_test)
print("Rapport de classification :")
print(classification_report(y_test, y_pred))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
