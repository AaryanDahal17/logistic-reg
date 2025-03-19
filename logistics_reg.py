import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Dataset is from: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

def load_data():

    df = pd.read_csv('data.csv')
    
    df = df.drop(['id', 'Unnamed: 32'], axis=1)
    
    # Converting diagnosis to binary (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    

    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    
    return X, y

def sigmoid(z):
    """Compute sigmoid function"""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    """
    Compute binary cross-entropy cost function
    Args:
        X: input features (m, n)
        y: labels (m,)
        w: weights (n,)
        b: bias scalar
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    cost = -1/m * np.sum(y * np.log(f_wb + 1e-15) + (1 - y) * np.log(1 - f_wb + 1e-15))
    return cost

def compute_gradient(X, y, w, b):
    """
    Compute gradients for logistic regression
    Returns dj_dw, dj_db (gradients for weights and bias)
    """
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    dj_dw = 1/m * np.dot(X.T, (f_wb - y))
    dj_db = 1/m * np.sum(f_wb - y)
    
    return dj_dw, dj_db

def train_logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Train logistic regression using gradient descent
    Returns final weights, bias, and cost history
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    costs = []
    
    for i in range(num_iterations):
        cost = compute_cost(X, y, w, b)
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db
        
        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return w, b, costs

def predict(X, w, b):
    """Make binary predictions using trained weights"""
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    return (f_wb >= 0.5).astype(int)

def plot_cost_history(costs):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title('Cost History')
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Cost')
    plt.show()



