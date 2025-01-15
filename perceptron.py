import numpy as np
import pandas as pd

# Load data from iris.txt
file_path = r'C:\Users\sahar\Downloads\iris.txt'
data = pd.read_csv(file_path, sep=" ", header=None)  # Adjust delimiter (space-separated)
data.columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Label']

# Use only second and third features for simplicity
features = data[['Feature2', 'Feature3']].values
labels = data['Label'].values

# Map class labels to binary values for Setosa and Versicolor
class_1, class_2 = "Iris-versicolor", "Iris-setosa"
mask = (labels == class_1) | (labels == class_2)
features = features[mask]
labels = labels[mask]

# Convert labels to +1 and -1
labels = np.where(labels == class_1, 1, -1)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Perceptron algorithm
def perceptron(X, y, max_iter=1000):
    w = np.zeros(X.shape[1])  # Initialize weight vector
    b = 0  # Initialize bias
    mistakes = 0  # Count the number of mistakes

    for _ in range(max_iter):
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += yi * xi
                b += yi
                mistakes += 1

    return w, b, mistakes

# Train the perceptron
weights, bias, total_mistakes = perceptron(X_train, y_train)

# Print results
print("Final weight vector:", weights)
print("Final bias:", bias)
print("Total mistakes during training:", total_mistakes)

# Compute accuracy on the test set
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

y_pred = predict(X_test, weights, bias)
accuracy = np.mean(y_pred == y_test)
print("Test set accuracy:", accuracy)
