# credits to video link: https://www.youtube.com/watch?v=mA5nwGoRAOo&index=19&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v#t=42.977805

import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

# build the KNN function
def k_nearest_neighbors (data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclindean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclindean_distance, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    #print("counter.votes is: ", Counter(votes).most_common(1))   
    vote_results = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][0] / k
    #print(vote_results, confidence)
    
    return vote_results, confidence


# load the data
df = pd.read_csv("D:/ML_Py/breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace = True)
df.drop('id', 1, inplace = True)
full_data = df.astype(float).values.tolist() # some data are imported as string, so make sure all the data are converted into integer or floats
random.shuffle(full_data)


# partition the data
test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))] # from begin to the last 20% data
test_data = full_data[-int(test_size * len(full_data)):] # last 20% data


# create a dictionary for the dataset to feed into the KNN function
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

for i in train_data:
    train_set[i[-1]].append(i[:-1]) # every thing except the 'class' value
    
for i in test_data:
    test_set[i[-1]].append(i[:-1])

# pass the data into the KNN model
correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors (train_set, data, k=5)
        if group == vote:
            correct += 1
        else: 
            print('confidence of incorrect:', confidence)
        total += 1
        
print('Accuracy:', correct/total)





