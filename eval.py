import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistics_reg import train_logistic_regression, predict

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    return X, y

def train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    m = len(y)
    indices = np.random.permutation(m)
    test_size = int(m * test_size)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize_features(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled

def compute_metrics(y_true, y_pred):
    true_pos = np.sum((y_true == 1) & (y_pred == 1))
    true_neg = np.sum((y_true == 0) & (y_pred == 0))
    false_pos = np.sum((y_true == 0) & (y_pred == 1))
    false_neg = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (true_pos + true_neg) / len(y_true)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_cost_history(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost History')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2))
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(int(cm[i, j])), 
                    ha='center', va='center')
    
    plt.show()

def main():
    print("Loading data...")
    X, y = load_data('data.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    print("\nTraining logistic regression...")
    w, b, costs = train_logistic_regression(X_train_scaled, y_train)
    
    y_pred_train = predict(X_train_scaled, w, b)
    y_pred_test = predict(X_test_scaled, w, b)
    
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    
    print("\nTraining Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    plot_cost_history(costs)
    plot_confusion_matrix(y_test, y_pred_test)

if __name__ == "__main__":
    main() 