from preprocess import load_and_preprocess_data
from model import create_svm_model, create_rf_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Charger les données
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# Définir les hyperparamètres à tester pour SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

# GridSearch pour SVM
print("Optimisation des hyperparamètres pour SVM...")
svm = GridSearchCV(create_svm_model(), param_grid_svm, cv=3, scoring='accuracy')
svm.fit(X_train, y_train)
print("Meilleurs paramètres SVM :", svm.best_params_)

# Sauvegarder le meilleur modèle SVM
os.makedirs('models', exist_ok=True)
joblib.dump(svm.best_estimator_, 'models/best_model.sav')

# Évaluation
y_pred = svm.best_estimator_.predict(X_test)
print("Classification Report SVM :")
print(classification_report(y_test, y_pred))
