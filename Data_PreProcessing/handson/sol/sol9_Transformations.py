# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:30:56 2022

@author: Megha
"""

import pandas as pd
import numpy as np 
import scipy.stats as stats
import pylab
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine import transformation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


calorie_df = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/calories_consumed.csv')
calorie_df.info()

#check for NaN
calorie_df.isna().sum()

#Identify duplicate records in the data
duplicate = calorie_df.duplicated()  

sum(duplicate)

#Exploratory Data Analysis for univariate data
calorie_df.mean()
calorie_df.median()
calorie_df.mode()

calorie_df.skew()
calorie_df.kurt()

calorie_df.describe()


#Normal Quantile-Quantile Plot

stats.probplot(calorie_df['Weight gained (grams)'], dist = "norm", plot = pylab)

stats.probplot(calorie_df['Calories Consumed'], dist = "norm", plot = pylab)

#Transformation to make 'Weight gained (grams)' variable normal
stats.probplot(np.log(calorie_df['Weight gained (grams)']), dist = "norm", plot = pylab)

#Transorm training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(calorie_df['Weight gained (grams)'])

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.distplot(calorie_df['Weight gained (grams)'], hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Non-Normal", color = "green", ax = ax[0])

sns.distplot(fitted_data, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 2},
             label = "Normal", color = "green", ax = ax[1])

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)

#YeoJohnson tranformer 
tf = transformation.YeoJohnsonTransformer(variables= 'Weight gained (grams)')

edu_tf = tf.fit_transform(calorie_df)

prob = stats.probplot(calorie_df['Weight gained (grams)'], dist = stats.norm, plot = pylab)

#Standardization and Normalization
scaler = StandardScaler()

df = scaler.fit_transform(calorie_df)
dataset = pd.DataFrame(df)
res = dataset.describe()


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(calorie_df)
b = df_norm.describe()

#MinMaxScaler
minmaxscale = MinMaxScaler()

calorie_minmax = minmaxscale.fit_transform(calorie_df)
df_calorie = pd.DataFrame(calorie_minmax)
minmax_res = df_calorie.describe()

#RobustScaler

robust_model = RobustScaler()
df_robust = robust_model.fit_transform(calorie_df)

dataset_robust = pd.DataFrame(df_robust)
res_robust = dataset_robust.describe()
