"""
Implementing Boosting using decision tree as base estimator
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split    
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset of Wine
df=pd.DataFrame(load_wine().data,columns=load_wine().feature_names)
X=df.values
y=load_wine().target

#preprocessing of data
df=df.fillna(df.mean())  # Handling missing values by filling with mean

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)  # Scaling the features

# Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# Decision Tree Classifier as base estimator
dt_model=DecisionTreeClassifier(random_state=42)

# AdaBoost Classifier with Decision Tree as base estimator
boosting_model=AdaBoostClassifier(estimator=dt_model,n_estimators=100,random_state=42)

# Train the model
boosting_model.fit(X_train,y_train)

# Save Model
joblib.dump(boosting_model, "model.joblib")

# Save Scaler
joblib.dump(scaler, "scaler.joblib")

print("Model Saved Successfully")


