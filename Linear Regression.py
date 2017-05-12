from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import style
import random

# generate a dataset
#x = np.transpose(np.matrix(np.random.normal(size = 500)))
#x = np.random.normal(size = 500)
#y = 8 * x + 3
#plt.scatter(x,y)

def get_slop_intercept(x, y):
    m = ( ((mean(x) * mean(y)) - mean(x * y)) / 
         (mean(x) **2 - mean (x **2 )) )
    b = mean(y) - mean(x) * m
    return m, b

def create_datasets(n, variance, step = 2, correlation=False):
    val = 1
    y = []
    for i in range(n):
        temp = val + random.randrange(-variance, variance)
        y.append(temp)
        if correlation and correlation == 'pos': 
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    x = [i for i in range(n)]
    return np.array(x, dtype = np.float64), np.array(y,dtype = np.float64)

# create the dataset
x, y = create_datasets(150, 100, 3, correlation = "neg")

m,b = get_slop_intercept(x,y)
print("m is", m)
print("b is", b)

regression_line = [(m*x) + b for x in x]
plt.scatter(x,y)
plt.plot(x, regression_line)
style.use("fivethirtyeight")

'''
model = LinearRegression()
model.fit(x, y)
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f", np.mean((model.predict(x - y) ** 2)))
'''
