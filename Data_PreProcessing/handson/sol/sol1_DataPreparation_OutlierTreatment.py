# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:20:53 2022

@author: Megha
"""
#Boston Housing Dataset

import pandas as pd
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer

boston_data = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/boston_data.csv')
boston_data.info()

#Exploratory Data Analysis for univariate data
#1st business moment - Measure of Central Tendency
boston_data.mean()
boston_data.median()
boston_data.mode()

res = boston_data.describe()

#2nd business moment - Measures of Dispersion
boston_data.var()
boston_data.std()

#3rd business moment - Skewness
boston_data.skew()

#4th business moment - Kurtosis (Measure of Peakedness)
boston_data.kurt()


#check for NaN
boston_data.isna().sum()

#Visualize numeric data using boxplot for outliers

boston_data.plot(kind='box', subplots=True, sharey= False, figsize = (25,14))
plt.subplots_adjust(wspace=0.75)
plt.show()

#column - chas is a charles river dummy variable encoded with 1 for tract bounds river and 0 otherwise
#We can notice outliers in column - crim, zn, rm, dis, ptratio, black, lstat, medv

#Use Winsorization function to treat outliers using capping method - IQR

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['crim'])

boston_data['crim'] = winsor.fit_transform(boston_data[['crim']])

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['zn'])

boston_data['zn'] = winsor.fit_transform(boston_data[['zn']])


winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['rm'])

boston_data['rm'] = winsor.fit_transform(boston_data[['rm']])

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['dis'])

boston_data['dis'] = winsor.fit_transform(boston_data[['dis']])

winsor = Winsorizer(capping_method='iqr', tail = 'both',fold=1.5, variables=['ptratio'])

boston_data['ptratio'] = winsor.fit_transform(boston_data[['ptratio']])

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['black'])

boston_data['black'] = winsor.fit_transform(boston_data[['black']])

winsor = Winsorizer(capping_method='iqr', tail = 'both',fold=1.5, variables=['lstat'])

boston_data['lstat'] = winsor.fit_transform(boston_data[['lstat']])

winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['medv'])

boston_data['medv'] = winsor.fit_transform(boston_data[['medv']])


#Again visualize the data 
boston_data.plot(kind='box', subplots=True, sharey= False, figsize = (25,14))
plt.subplots_adjust(wspace=0.75)
plt.show()












