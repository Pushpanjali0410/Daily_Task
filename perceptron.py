import numpy as np

# AND gate dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0

learning_rate = 0.1
epochs = 10

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        prediction = step_function(linear_output)

        error = y[i] - prediction

        weights += learning_rate * error * X[i]
        bias += learning_rate * error

print("Final Weights:", weights)
print("Final Bias:", bias)

# Testing
print("\nPredictions:")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    prediction = step_function(linear_output)
    print(f"Input: {X[i]}, Predicted: {prediction}, Actual: {y[i]}")