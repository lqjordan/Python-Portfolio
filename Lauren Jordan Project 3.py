#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:18:44 2021

@author: jordanl00
"""

#%%
#Part 1
import pandas as pd
wine = pd.read_csv('red wine quality.csv')

#Part 2
print(wine.shape)
print(wine.columns)
print(wine.head(10))

#Part 3
print (wine.describe())

#Part 4
wine = wine.dropna()

#Part 5
import matplotlib.pyplot as plt
wine.hist(figsize = (8,10))
plt.savefig('histogram.png')
plt.show()

#Part 6
wine.plot.scatter(x='pH', y='alcohol',c='quality')
plt.savefig('scatter plot.png')
plt.show()

#Part 7
def quality_level(q):
    if 3>= q <= 4:
        return 'poor'
    elif 5>= q <=6:
        return 'fair'
    elif 7>= q <=8:
        return'good'

#Part 8
l = map(quality_level, wine.quality)
wine_qual = list(l)
print (wine_qual)
wine['quality_level']=wine_qual
print (wine.columns)

#Part 9
wine_quality = wine.quality_level.value_counts()
plt.figure()
wine_quality.plot.pie(figsize=(10,10),autopct='%1.2f%%',legend=False)
plt.savefig('quality pie chart.png')
plt.show()

#Part 10
GroupByQuality = wine.groupby('quality_level').mean()
print(GroupByQuality)
plt.figure()
GroupByQuality.plot.bar(rot = 45)
plt.savefig ('bar chart.png')
plt.show()

#Part 11
wine = wine.drop('quality_level',axis = 1)
print (wine.columns)

#Part 12
correlations = wine.corr()
correlations.to_csv('correlation.csv')
pd.plotting.scatter_matrix(wine,alpha = 0.4, figsize = (30,30))
plt.savefig('scatter matrix.png')
plt.show()

#Part 13
from scipy import stats
stat, p = stats.normaltest(wine.density)
alpha = 0.05
if p > alpha:
    print ('The p-value is {0:.8f}, which is > 0.05 (alpha/significanc level). Distribution is normal, fail to reject H0.'.format (p))
else:
    print ('The p-value is {0:.8f}, which is <= 0.05 (alpha/significanc level). Distribution is not normal, so reject H0.'.format (p))

#Part 14
qualitycorrelation = correlations[(abs(correlations.quality)<0.15)]
#finds all variables whose correlation absolute value with quality is less than 0.15
print(qualitycorrelation.index)
qc = list (qualitycorrelation.index)
for i in qc:
    wine = wine.drop(labels=i,axis=1)
#drops all variables whose correlation absolute value is less than 0.15 from wine dataframe
print(wine.columns)
print(wine.shape)

#Part 15
y = wine.quality
x = wine.drop ('quality', axis = 1)

#Part 16
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.25, \
                                                     random_state = 2000)
#Part 17
from sklearn.linear_model import LinearRegression
wine_quality = LinearRegression()
wine_quality.fit (x_train, y_train)
for i, col_name in enumerate (x.columns):
    print ("The coefficient for {} is {}".format(col_name, wine_quality.coef_[i]))
print ('The intercept for wine_quality is ', wine_quality.intercept_)

#Part 18
r_squared = wine_quality.score(x_test, y_test)
pv = len(wine_quality.coef_)
n = len (y_test)
adjusted_r_squared = r_squared-(1-r_squared)*p/(n-p-1)
from sklearn.metrics import mean_squared_error
y_predict = wine_quality.predict(x_test)
mse = mean_squared_error (y_predict, y_test)
import numpy as np
rmse = np.sqrt(mse)
print ('The R^2 value is {},\
       the adjusted R^2 is {},\
           the root mean square error (RMSE) is {}'.format(r_squared,adjusted_r_squared,rmse))

#Part 19
z_predict = wine_quality.predict ([[0.75, 0, 100, 0.995, 0.85, 10 ]])
z_predict = float(z_predict)
hi = round(z_predict) + float(rmse)
high = round(hi)
lo = round(z_predict) - float(rmse)
low = round(lo)
new_rmse= float(rmse)
print ('The wine quality is predicted to be {}, which is {}.'.format (z_predict, quality_level(z_predict)))
print ('Since the root mean square error is about {},\
       the actual quality should fall somewhere between {} and {}.'.format(round(new_rmse),low,high))

#Part 20
data = {'R squared':r_squared,'adjusted R squared':adjusted_r_squared,\
        'root_mean_square_error':rmse,'predicted value':y_predict}
predicted_value = pd.DataFrame(data,columns = ['R squared','adjusted R squared',\
                                               'root_mean_square_error','predicted value'])
predicted_value.to_csv('predicted value.csv')
print(predicted_value)



