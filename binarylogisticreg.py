#importing Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the explicit Train and Test datasets
print("Loading datasets...")
df_train = pd.read_csv(r"dataset\train.csv")
df_test = pd.read_csv(r"dataset\test.csv")

# 2. Extract text data and handle any potential missing values safely
X_train_raw = df_train["text"].astype(str)
X_test_raw = df_test["text"].astype(str)

# 3. Fit the LabelEncoder on training targets and transform both sets
# This converts labels (like 'pos'/'neg' or '1'/'0') into clean binary 1s and 0s
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train["label"]) #fit on training labels to learn the mapping, then transform to get numeric labels
y_test = label_encoder.transform(df_test["label"]) #transform test labels using the same mapping learned from training labels to ensure consistency

print(f"Training classes detected: {label_encoder.classes_}")

# 4. Convert text to numbers using TF-IDF Vectorization
# We FIT on the training text, and only TRANSFORM the test text to avoid data leakage
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# 5. Initialize and Train the Logistic Regression Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Make Predictions on the independent test set
y_pred = model.predict(X_test)

# 7. Print Performance Metrics
print("\n=== MODEL PERFORMANCE METRICS ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in label_encoder.classes_]))

print("Confusion Matrix:")  
print(confusion_matrix(y_test, y_pred))
