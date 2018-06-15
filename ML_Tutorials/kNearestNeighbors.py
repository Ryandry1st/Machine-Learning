#builds up a K nearest neighbors classifier
from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclid_dist = np.linalg.norm(np.array(features-np.array(predict)))
            distances.append([euclid_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_results = Counter(votes).most_common(1)[0][0]
        
    return vote_results

df = pd.read_csv('breast-cancer-wisconsin-data.txt')
df = df.replace('?', pd.np.nan).dropna(axis=0, how='any')

df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print ('Accuracy:', float(correct)/float(total))



    
