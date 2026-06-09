import pickle
from sklearn.datasets import load_iris

# Load Model
with open("models/iris_model.pkl", "rb") as file:
    model = pickle.load(file)

iris = load_iris()

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("Predicted Flower:")
print(iris.target_names[prediction[0]])