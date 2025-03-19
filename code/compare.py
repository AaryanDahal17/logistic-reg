import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistics_reg import train_logistic_regression, predict, compute_cost
from dummyclassifier import train_dummy_classifier, get_dummy_predictions, compute_dummy_metrics
from main import load_data, train_test_split, standardize_features, compute_metrics

def compare_models():
    # Load and prepare data
    print("Loading data...")
    X, y = load_data('data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)

    # Train and evaluate logistic regression model
    print("\nTraining logistic regression model...")
    w, b, costs = train_logistic_regression(X_train_scaled, y_train)
    y_pred_log_train = predict(X_train_scaled, w, b)
    y_pred_log_test = predict(X_test_scaled, w, b)
    
    log_train_metrics = compute_metrics(y_train, y_pred_log_train)
    log_test_metrics = compute_metrics(y_test, y_pred_log_test)

    # Train and evaluate dummy classifier
    print("\nTraining dummy classifier...")
    dummy_model = train_dummy_classifier(X_train_scaled, y_train)
    y_pred_dummy_train = get_dummy_predictions(dummy_model, X_train_scaled)
    y_pred_dummy_test = get_dummy_predictions(dummy_model, X_test_scaled)
    
    dummy_train_metrics = compute_dummy_metrics(y_train, y_pred_dummy_train)
    dummy_test_metrics = compute_dummy_metrics(y_test, y_pred_dummy_test)

    # Print comparison
    print("\n=== Model Comparison ===")
    print("\nLogistic Regression Results:")
    print("Training Metrics:")
    for metric, value in log_train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nTest Metrics:")
    for metric, value in log_test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nDummy Classifier Results:")
    print("Training Metrics:")
    for metric, value in dummy_train_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nTest Metrics:")
    for metric, value in dummy_test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot comparison
    plot_metric_comparison(
        log_test_metrics,
        dummy_test_metrics,
        "Model Performance Comparison (Test Set)"
    )

def plot_metric_comparison(metrics1, metrics2, title):
    """Plot bar chart comparing metrics of both models"""
    metrics = list(metrics1.keys())
    model1_values = [metrics1[metric] for metric in metrics]
    model2_values = [metrics2[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, model1_values, width, label='Logistic Regression')
    rects2 = ax.bar(x + width/2, model2_values, width, label='Dummy Classifier')

    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_models() 