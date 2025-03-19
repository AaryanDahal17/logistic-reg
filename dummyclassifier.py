import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    return X, y

def train_dummy_classifier(X_train, y_train, strategy='stratified'):
    """
    Train a dummy classifier using specified strategy
    Returns trained model
    """
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    dummy_clf.fit(X_train, y_train)
    return dummy_clf

def get_dummy_predictions(model, X):
    """Make predictions using dummy classifier"""
    return model.predict(X)

def compute_dummy_metrics(y_true, y_pred):
    """Compute various metrics for the dummy classifier"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    } 