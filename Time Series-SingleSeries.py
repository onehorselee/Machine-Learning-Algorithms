'''Outline: 
# how to load and format a single time series data
# how to index and plot the time series data '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series

# two ways to import the time series data: data == data3, but data != data2
data = Series.from_csv('D:/ML_Py/AirPassengers.csv', header = 0)
data2 = pd.read_csv('D:/ML_Py/AirPassengers.csv', header = 0, parse_dates = [0])
data3 = pd.read_csv('D:/ML_Py/AirPassengers.csv', header = 0, parse_dates = [0], index_col = 0)

# select data
data['1949'] # all the data in the year of 1949
data['1949-01'] # all the data in the Jan. of 1949

# four ways to plot the time series
data.plot()

data2.plot(y = '#Passengers', x = 'Month')

plt.plot(data2['Month'], data2['#Passengers'])
plt.show()

plt.plot(data2.iloc[:,0], data2.iloc[:,1])
plt.show()

# plot the data within a time range or a value range
data['1951': '1959'].plot()
data['1951-12': '1959-08'].plot()
data[data>300].plot()
