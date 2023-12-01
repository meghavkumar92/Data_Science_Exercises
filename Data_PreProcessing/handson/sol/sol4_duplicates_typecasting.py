# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:19:42 2022

@author: Megha
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/OnlineRetail.csv', encoding= 'unicode_escape')
data.info()
data.dtypes

#check for NaN
data.isna().sum()

#Q1
#Missing Values - Imputation
#CustomerID is a unique value and as we have 135080 rows of nan. We can replace nan with some constant.
#Constant Value Imputer (by default nan is replaced by zero)

constant_imputer1 = SimpleImputer(missing_values = np.nan, strategy= 'constant')
data["CustomerID"] = pd.DataFrame(constant_imputer1.fit_transform(data[["CustomerID"]]))
data.isna().sum()

data.info()

#Type casting
#Column CustomerID is a unique key and should always be integer. So typecast float to int
data.CustomerID = data.CustomerID.astype('int64')
data.dtypes

#Description column has missing product details. We can replace such rows with any constant.
#Constant Value Imputer (replacing nan with string 'Not defined')
constant_imputer2 = SimpleImputer(missing_values = np.nan, strategy= 'constant', fill_value= "NOT DEFINED")

data["Description"] = pd.DataFrame(constant_imputer2.fit_transform(data[["Description"]]))
data.isna().sum()

data.info()


#Q2
#Duplicate handling

d = data.duplicated()

sum(d)

#removing duplicates

data1 = data.drop_duplicates()

#check if duplicates are removed
dnew = data1.duplicated()
sum(dnew)

#Q3
#Exploratory data analysis

data1.Quantity.mean()  #average quantity purchased
data1.Quantity.median() 
data1.Quantity.mode() 

data1.UnitPrice.mean() #average unit price of items
data1.UnitPrice.median()
data1.UnitPrice.mode()

data1.UnitPrice.var()
data1.UnitPrice.std()

data1.Quantity.var()
data1.Quantity.std()

data1.UnitPrice.skew()
data1.Quantity.skew() #-ve 

data1.UnitPrice.kurt()
data1.Quantity.kurt()


range_q= max(data1.Quantity) - min(data1.Quantity)
range_q

range_u= max(data1.UnitPrice) - min(data1.UnitPrice)
range_u

#Data Visualization

#boxplot
plt.figure()
plt.boxplot(data1.Quantity)

plt.boxplot(data1.UnitPrice)

#Outlier Removal 
IQR = data1['UnitPrice'].quantile(0.75) - data1['UnitPrice'].quantile(0.25)

lower_limit = data1['UnitPrice'].quantile(0.25)-(IQR*1.5)
upper_limit = data1['UnitPrice'].quantile(0.75)+(IQR*1.5)

outliers_data = np.where(data1.UnitPrice > upper_limit, True, np.where(data1.UnitPrice < lower_limit, True, False))
data_trimmed = data1.loc[~(outliers_data), ]
data1.shape, data_trimmed.shape

sns.boxplot(data_trimmed.UnitPrice)

IQR_1 = data1['Quantity'].quantile(0.75) - data1['Quantity'].quantile(0.25)

lower_limit = data1['Quantity'].quantile(0.25)-(IQR_1*1.5)
upper_limit = data1['Quantity'].quantile(0.75)+(IQR_1*1.5)

outliers_dat = np.where(data1.Quantity > upper_limit, True, np.where(data1.Quantity < lower_limit, True, False))
data_trimmed = data1.loc[~(outliers_dat), ]
data1.shape, data_trimmed.shape

sns.boxplot(data_trimmed.Quantity)


#histogram
plt.hist(data_trimmed.Quantity)
plt.hist(data_trimmed.Quantity,bins = [-50, -30, -10, 10, 30, 50])
plt.hist(data_trimmed.UnitPrice)


#scatter plot
plt.scatter(x = data_trimmed['Quantity'], y = data_trimmed['Country']) 

data1.head()
sns.boxplot(data1.UnitPrice)
