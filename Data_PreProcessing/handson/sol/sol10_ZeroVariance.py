# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:32:39 2022

@author: Megha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dt_df = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/Z_dataset.csv')

#check for NaN
dt_df.isna().sum()

#Identify duplicate records in the data
duplicate = dt_df.duplicated()  

sum(duplicate)

dt_df = dt_df.drop(['Id'], axis=1)

# Exploratory Data Analysis
dt_df.mean()
dt_df.median()
dt_df.mode()

dt_df.skew()
dt_df.kurt()

dt_df.describe()

#copy colour column to a new dataframe and drop the column in dt_df(has constant value 'Blue', 'Green', 'Orange')
dt_new = dt_df['colour']

dt_df = dt_df.drop(['colour'], axis=1)

dt_df.var()
dt_df.var() == 0
dt_df.var(axis=0) == 0


#Data Visualization

plt.hist(dt_df['square.length'])

sns.displot(dt_df['square.length'])
sns.kdeplot(dt_df['square.length'])

sns.kdeplot(dt_df['square.breadth'])

sns.kdeplot(dt_df['rec.breadth'])

sns.kdeplot(dt_df['rec.Length']) 
