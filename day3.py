# performance metrics for iris dataset from scratch and compare to scikit-learn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
data = load_iris()
x=data.data
y=data.target

# Split the dataset into training and testing sets
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=10000)
model.fit(X_train,Y_train)

y_pred=model.predict(X_test)

TP=np.sum((y_pred==1)&(Y_test==1))
TN=np.sum((Y_test==0)&(y_pred==0))
FP=np.sum((Y_test==0)&(y_pred==1))
FN=np.sum((Y_test==1)&(y_pred==0))

accuracy=(TP+TN)/(TP+TN+FP+FN) # how many actually we predicted correctly
precision=TP/(TP+FP) if TP+FP!=0 else 0  # how many of the predicted positives are actually positive
recall=TP/(TP+FN) if TP+FN!=0 else 0 # how many of the actual positives are correctly predicted
f1=2*precision*recall/(precision+recall) if precision+recall!=0 else 0 # harmonic mean of precision and recall

acc_sc=accuracy_score(y_pred,Y_test)
prec_sc=precision_score(y_pred,Y_test,average='weighted')
recall_sc=recall_score(y_pred,Y_test,average='weighted')
f1_sc=f1_score(y_pred,Y_test,average='weighted')

print(f"Custom Metrics: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1:.2f}")
print(f"Scikit-learn Metrics: Accuracy={acc_sc:.2f}, Precision={prec_sc:.2f}, Recall={recall_sc:.2f}, F1 Score={f1_sc:.2f}")    
