#Libraries using in this project
import pandas as pd

#Load the dataset
dataset = pd.read_csv('cleaned_breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Training the Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test) 

#Confusing Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import joblib


joblib.dump(classifier, 'classifier.pkl') 