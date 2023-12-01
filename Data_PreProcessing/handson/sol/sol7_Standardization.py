# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:15:02 2022

@author: Megha
"""

#STANDARDIZATION & NORMALIZATION

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


seeds_df = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/Seeds_data.csv')
seeds_df.info()

#check for NaN
seeds_df.isna().sum()

#Identify duplicate records in the data
duplicate = seeds_df.duplicated()  
sum(duplicate)

# visualize the data 
seeds_df.plot(kind='box', subplots=True, sharey= False, figsize = (20,8))
plt.subplots_adjust(wspace=0.5)
plt.show()

res = seeds_df.describe()
res

#Standardization
scaler = StandardScaler()

df = scaler.fit_transform(seeds_df)
std_df = pd.DataFrame(df)
res = std_df.describe()

#Normalization

#custom function 
def norm_func(i):
    x = (i - i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(seeds_df)
res_nd = df_norm.describe()

#alternative method
minmaxscale = MinMaxScaler()

minmax_data = minmaxscale.fit_transform(seeds_df)
minmax_df = pd.DataFrame(minmax_data)
minmax_res = minmax_df.describe()

#Robust Scaling

robust_model = RobustScaler()

robust_data = robust_model.fit_transform(seeds_df)

robust_df = pd.DataFrame(robust_data)

res_robust = robust_df.describe()


