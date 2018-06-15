"""K nearest neighbors.
Probably should be an odd number of K so that there cannot be a tie
Groups applications by distance to other points."""
#we will do Euclidian distance, which is a super slow algorithm for large data
#can be threaded decently but still is slow

#uses breast cancer data from https://archive.ics.uci.edu/ml/datasets.html

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
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

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measure = np.array([[10,2,1,4,1,2,3,2,1], [6,3,5,7,8,4,2,4,7]])
example_measure = example_measure.reshape(len(example_measure),-1)

prediction = clf.predict(example_measure)
print prediction
