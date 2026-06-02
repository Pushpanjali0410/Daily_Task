import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (important for perceptron)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize weights and bias
weights = np.zeros(X_train.shape[1])
bias = 0

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(X_train[i], weights) + bias
        prediction = step_function(linear_output)

        error = y_train[i] - prediction

        weights += learning_rate * error * X_train[i]
        bias += learning_rate * error

print("Final Weights:")
print(weights)

print("\nFinal Bias:")
print(bias)

# Testing
predictions = []

for x in X_test:
    linear_output = np.dot(x, weights) + bias
    prediction = step_function(linear_output)
    predictions.append(prediction)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Display first 10 predictions
print("\nSample Predictions:")
for i in range(10):
    print(f"Actual: {y_test[i]}, Predicted: {predictions[i]}")