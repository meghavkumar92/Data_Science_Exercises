# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:09:29 2022

@author: Megha
"""
import pandas as pd

iris_data = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/iris.csv')
iris_data.info()

iris_data.columns

# Remove column name 'Unnamed: 0'
iris_data = iris_data.drop(['Unnamed: 0'], axis=1)

iris_data.describe()

#iris_data['Sepal.Length'].unique()

#Converting the continuous value of sepal length column data into discrete value as 'short' and 'long'(categorical)
iris_data['Sepal_len_new'] = pd.cut(iris_data['Sepal.Length'],
          bins = [ min(iris_data['Sepal.Length']), iris_data['Sepal.Length'].mean(), max(iris_data['Sepal.Length']) ], 
          include_lowest = True, 
          labels = ['short', 'long'])

iris_data['Sepal_len_new'].value_counts()

# iris_data['Sepal.Width'].unique()

#Converting the continuous value of sepal width column data into discrete value as 'narrow' and 'broad'(categorical)
iris_data['Sepal_width_new'] = pd.cut(iris_data['Sepal.Width'],
                                      bins = [min(iris_data['Sepal.Width']),iris_data['Sepal.Width'].mean() ,max(iris_data['Sepal.Width'])],
                                      include_lowest = True,
                                      labels = ['narrow',' broad'])


iris_data['Sepal_width_new'].value_counts()

