from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def create_svm_model(C=1.0, kernel='rbf', gamma='scale'):
    return SVC(C=C, kernel=kernel, gamma=gamma)

def create_rf_model(n_estimators=100, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
