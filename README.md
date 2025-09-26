# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Import the necessary python packages

Step 3. Read the dataset.

Step 4. Define X and Y array.

Step 5. Define a function for costFunction,cost and gradient.

Step 6. Define a function to plot the decision boundary and predict the Regression value

Step 7. End

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANDHIYA M
RegisterNumber:  212224220086
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("place1.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return  -np.sum(y*np.log(h)+(1-y)*log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print('Accuracy:',accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

Accuracy

<img width="311" height="31" alt="491896642-4edda0fe-4760-458c-b41f-b108bd7e89d6" src="https://github.com/user-attachments/assets/5cbf4ebf-60d3-4014-8acb-bf577290da46" />

y_pred:

<img width="735" height="132" alt="491896599-14bc65e5-bf41-48cd-ac30-7c84fab48d20" src="https://github.com/user-attachments/assets/75d12361-5428-45b5-8db1-b13f219498d3" />

Y:

<img width="769" height="132" alt="491896693-8e86e51e-8c79-4519-b93d-a8f83802d67e" src="https://github.com/user-attachments/assets/26cf5935-cd42-44f4-bbd4-df8e3b067b18" />

Y_Prednew:

<img width="43" height="26" alt="491896759-5b290b48-75e4-40cc-a75d-f3885557ffd4" src="https://github.com/user-attachments/assets/251f1cc1-c642-4d3a-b215-74aacbb1531b" />





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

