import pandas as pd
import numpy as np

def load_and_preprocess_data():
    train_df = pd.read_csv('data/fashion-mnist_train.csv')
    test_df = pd.read_csv('data/fashion-mnist_test.csv')
    
    # Séparer les features (X) et les labels (y)
    X_train = train_df.iloc[:, 1:].values / 255.0
    y_train = train_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values / 255.0
    y_test = test_df.iloc[:, 0].values
    
    return X_train, y_train, X_test, y_test
