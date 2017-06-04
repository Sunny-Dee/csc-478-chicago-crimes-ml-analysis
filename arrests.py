#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:19:11 2017

@author: delianaescobari
"""
"""
Is there justice in this world? We want to know that given a specific type
of crime, if we can tell whether or not there will be an arrest.

For this model we will at two types of crime we are interested particularly 
interested: as thefts and assault. 
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os

mydir = "/Users/delianaescobari/Google Drive/CSC-478-MachineAlgos/group project"
os.chdir(mydir)
data_filename = "Crimes_-_2016.csv"

dataset = pd.read_csv(data_filename)

## Turn date column into datetime and just get the month
#dates = pd.to_datetime(dataset['Date'], format="%m/%d/%Y %H:%M:%S PM")
def convert_time(dates):
    time = []
    for i in range(dates.shape[0]):
        time.append(dates.iloc[i].split(" ")[0])
        
    return time


def get_hours(dates):
    hours = []
    for i in range(dates.shape[0]):
        h = dates.iloc[i].split(" ")
        am_pm = h[2]
        
        hour = int(h[1].split(":")[0])
        
        if (am_pm is "PM"):
            hour = hour + 12
            
        hours.append(hour)
    return hours
        

dates = convert_time(dataset['Date'])
hours = get_hours(dataset['Date'])
dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
dataset['Hour'] = pd.to_numeric(hours)

## Drop noisy variables and redundant variables: description, updated on, beat, year 
dataset = dataset.drop('Description', 1)
dataset = dataset.drop('Updated On', 1)
dataset = dataset.drop('Beat', 1)
dataset = dataset.drop('Year', 1) ## all in 2016
dataset = dataset.drop('Block', 1) 
dataset = dataset.drop('Date', 1) ## only using month 
dataset = dataset.drop('Location', 1) ## already have lat and long as separate fields
dataset = dataset.drop('X Coordinate', 1) ##already have lat
dataset = dataset.drop('Y Coordinate', 1) ## already have long


## Drop UICR because it's not independent of what we are trying to predict
dataset = dataset.drop('IUCR', 1)
dataset = dataset.drop('FBI Code', 1) ## not independent variable

# Drop na values. We will use missing rows in other analysis
dataset = dataset.dropna(axis=0, how='any')

## Uncomment if you'd like to filter to keep only theft and assault
dataset = dataset[(dataset['Primary Type'] == 'THEFT') | (dataset['Primary Type'] == 'ASSAULT')]

#Split values between dependent variable and independent variables
X1 = dataset.iloc[:, 2:4]
X2 = dataset.iloc[:, 5:]

X = pd.concat([X1, X2], axis=1, join='inner')
X = X.values
y = dataset.iloc[:, 4].values

# Clean categorical variables.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode Categorical variables 
primary_enc = LabelEncoder()
location_enc = LabelEncoder()
domestic_enc = LabelEncoder()

X[:, 0] = primary_enc.fit_transform(X[:, 0]) 
X[:, 1] = location_enc.fit_transform(X[:, 1]) 
X[:, 2] = domestic_enc.fit_transform(X[:, 2])

#Create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 2, 3, 4, 5])
X = onehotencoder.fit_transform(X).toarray()

#Split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Normalize variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# Get a sense of how much variance is explained by the number of variables
def variance_explained(variances, start_printing_at):   
    result = 0
    for i in range(len(variances)):
        result += variances[i]
        if (result > start_printing_at):
            print("{} variables explain {:.2f} variance".format(i, result))


variance_explained(explained_variance, 0.9)

# We decide to use 194 variables that explain 97% of the variance
pca = PCA(n_components = 194)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

## Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predict test results
y_pred = classifier.predict(X_test)


"""
Update:
    Improve our model with cross validation
"""
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()



#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Visualize cm matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(cm, ["Arrest", "No arrest"], title='Confusion matrix')


plt.show()


