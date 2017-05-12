"""
Created on Fri May 12 11:00:02 2017
@author: onehorselee
datasource: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
"""
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv("D:/ML_Py/breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True) # drop the "id"

x = np.array(df.drop('class', 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split (x,y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print('accuracy:', accuracy)

example_measures = np. array([4,2,1,1,1,2,3,2,1])
example_measures = example_measures.reshape(1,-1)
#example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
