import argparse
import csv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to sys.path to import LassoHomotopy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.LassoHomotopy import LassoHomotopy

DEFAULT_FILE = "small_test.csv"  # Default dataset

def load_data(file_path):
    """ Load and process data from the CSV file. """
    data = []
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    return X, y

def shuffle_data(X, y):
    """ Shuffle the dataset to prevent bias from ordered data. """
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def normalize_features(X):
    """ Normalize features to zero mean and unit variance. """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def cross_validate(X_train, y_train, alphas, kcf=10, max_iter=2000, tol=1e-4):
    """ Perform cross-validation to find the optimal alpha. """
    segment_length = len(X_train) // kcf

    def compute_mse(alpha, fold):
        X_train_segment = np.concatenate((X_train[:fold*segment_length], X_train[(fold+1)*segment_length:]), axis=0)
        y_train_segment = np.concatenate((y_train[:fold*segment_length], y_train[(fold+1)*segment_length:]), axis=0)
        X_validation_subset = X_train[fold*segment_length:(fold+1)*segment_length]
        y_validation_subset = y_train[fold*segment_length:(fold+1)*segment_length]

        model = LassoHomotopy(alpha=alpha, max_iter=max_iter, tol=tol, verbose=False)
        model.fit(X_train_segment, y_train_segment)
        predicted_y_values = model.predict(X_validation_subset)
        mse = np.mean((y_validation_subset - predicted_y_values) ** 2)
        return mse

    results = [(alpha, np.mean([compute_mse(alpha, fold) for fold in range(kcf)])) for alpha in alphas]
    
    return min(results, key=lambda x: x[1])

def test_predict(file_path, kcf, max_iter, tol):
    X, y = load_data(file_path)

    # Shuffle and normalize data
    X, y = shuffle_data(X, y)
    X = normalize_features(X)

    # Train-test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Define hyperparameter search space
    alphas = np.logspace(-4, 4, 20)

    # Cross-validation
    best_alpha, best_mse = cross_validate(X_train, y_train, alphas, kcf=kcf, max_iter=max_iter, tol=tol)
    print(f"Optimal Alpha: {best_alpha}, MSE: {best_mse:.4f}")

    # Train final model
    final_model = LassoHomotopy(alpha=best_alpha, max_iter=max_iter, tol=tol, verbose=True)
    final_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = final_model.predict(X_train)
    y_pred_test = final_model.predict(X_test)

    # Compute MSE
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(12, 5))

    # Training set plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.6, color='blue')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Training Set: Predicted vs Actual")

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Test Set: Predicted vs Actual")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lasso Regression with Homotopy Algorithm")
    parser.add_argument("file_path", nargs="?", default=DEFAULT_FILE, type=str, help="Path to the CSV file (default: test2.csv)")
    parser.add_argument("--kcf", type=int, default=10, help="Number of folds for cross-validation (default: 10)")
    parser.add_argument("--max_iter", type=int, default=2000, help="Maximum iterations for model training (default: 2000)")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for convergence (default: 1e-4)")
    
    args = parser.parse_args()
    test_predict(args.file_path, args.kcf, args.max_iter, args.tol) 
