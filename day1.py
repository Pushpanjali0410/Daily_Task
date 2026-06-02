import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv(r"dataset\Ecommerce_Sales_Prediction_Dataset.csv")

# 2. Preprocess Categorical Features (Convert text to numbers using One-Hot Encoding)
# This handles "Product_Category" and "Customer_Segment" automatically
df_encoded = pd.get_dummies(df, columns=["Product_Category", "Customer_Segment"], drop_first=True)

# 3. Define Features (X) and Target (y)
# We drop 'Units_Sold' (target) and 'Date' (linear regression cannot read raw date strings)
X = df_encoded.drop(columns=["Units_Sold", "Date"])
y = df_encoded["Units_Sold"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score (Accuracy Metric): {r2:.2f}")


