# SVM which is a binary classifier.
# Tries to create a hyper plane which seperates the two groups
# Finds the largest perpendicular distance to the nearest points from each group
# Theoretically should be linear data

import numpy as np
from sklearn import preprocessing, model_selection, svm
import pandas as pd
import random

df = pd.read_csv('breast-cancer-wisconsin-data.txt')
df = df.replace('?', pd.np.nan).dropna(axis=0, how='any')
#drops missing data

#df.replace('?', -99999, inplace=True)
#changes missing data to be clear outliers

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

accuracies = []
for i in range(500):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)
    
"""print(accuracy)

example_measure = np.array([[10,2,1,4,1,2,3,2,1], [5,3,5,6,7,4,2,4,7]])
example_measure = example_measure.reshape(len(example_measure),-1)

prediction = clf.predict(example_measure)
print prediction"""

print(sum(accuracies)/len(accuracies))
