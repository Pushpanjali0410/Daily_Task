## Logistic Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df= pd.read_csv(r"dataset\Ecommerce_Sales_Prediction_Dataset.csv")
# Preprocess Categorical Features (Convert text to numbers using One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=["Product_Category", "Customer_Segment"], drop_first=True)

x= df_encoded.drop(columns=["Units_Sold", "Date"])
y= df_encoded["Units_Sold"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred= model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
