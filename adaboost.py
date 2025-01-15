import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split

# Load the iris.txt dataset
file_path = r'C:\Users\sahar\Downloads\iris.txt'  # Update with your actual file path
data = pd.read_csv(file_path, sep=" ", header=None)
data.columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label']

# Filter for Versicolor and Virginica
data = data[(data['Label'] == "Iris-versicolor") | (data['Label'] == "Iris-virginica")]
data['Label'] = np.where(data['Label'] == "Iris-versicolor", 1, -1)

# Use only Feature2 and Feature3
X = data[['Feature2', 'Feature3']].values
y = data['Label'].values


def line_hypothesis(x1, x2, samples):
    """
    Compute the weak classifier defined by the line passing through points x1 and x2.
    Classifies points in 'samples' as +1 if above the line, -1 if below.
    """
    if x2[0] == x1[0]:  # Vertical line
        return np.sign(samples[:, 0] - x1[0])

    slope = (x2[1] - x1[1]) / (x2[0] - x1[0])
    intercept = x1[1] - slope * x1[0]

    return np.sign(samples[:, 1] - (slope * samples[:, 0] + intercept))


def compute_errors(X_train, y_train, X_test, y_test, classifiers, alphas):
    """
    Compute empirical and true errors for the combined classifiers.
    """
    train_errors = []
    test_errors = []
    for k in range(1, len(classifiers) + 1):
        H_train = np.sign(sum(alpha * clf for alpha, clf in zip(alphas[:k], classifiers[:k])))
        H_test = np.sign(sum(alpha * clf for alpha, clf in zip(alphas[:k], classifiers[:k])))

        train_errors.append(np.mean(H_train != y_train))
        test_errors.append(np.mean(H_test != y_test))

    return train_errors, test_errors


# Adaboost Algorithm
def adaboost(X, y, max_rounds=8):
    n = len(y)
    weights = np.ones(n) / n
    classifiers = []
    alphas = []

    for _ in range(max_rounds):
        best_error = float('inf')
        best_hypothesis = None

        for (i, j) in combinations(range(len(X)), 2):
            h = line_hypothesis(X[i], X[j], X)
            error = np.sum(weights * (h != y))
            if error < best_error:
                best_error = error
                best_hypothesis = h

        alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        weights *= np.exp(-alpha * y * best_hypothesis)
        weights /= np.sum(weights)

        classifiers.append(best_hypothesis)
        alphas.append(alpha)

    return classifiers, alphas


# Perform 100 runs
train_errors_avg = np.zeros(8)
test_errors_avg = np.zeros(8)

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=None)
    classifiers, alphas = adaboost(X_train, y_train)
    train_errors, test_errors = compute_errors(X_train, y_train, X_test, y_test, classifiers, alphas)

    train_errors_avg += np.array(train_errors)
    test_errors_avg += np.array(test_errors)

# Average the results
train_errors_avg /= 100
test_errors_avg /= 100

# Print the results
print("Final Results After 100 Runs:")
print("Training Errors (Averaged):")
for k, error in enumerate(train_errors_avg, 1):
    print(f"H_{k}: {error:.4f}")

print("\nTest Errors (Averaged):")
for k, error in enumerate(test_errors_avg, 1):
    print(f"H_{k}: {error:.4f}")
