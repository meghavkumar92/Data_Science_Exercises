# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 20:47:53 2022

@author: Megha
"""

#Missing values and Imutation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

claim_df =  pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/claimants.csv')
claim_df.info()

#check for NaN
claim_df.isna().sum()

# CLMSEX, CLMINSUR, SEATBELT are discrete data and can be imputed with mode/most frequent value
# CLMAGE has missing value/nan and it is a numeric data and can be imputed with mean/median

claim_df.CLMSEX.mode() #1

claim_df.CLMINSUR.mode() #1

claim_df.SEATBELT.mode() #0

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

claim_df['CLMSEX'] = pd.DataFrame(mode_imputer.fit_transform(claim_df[['CLMSEX']]))
claim_df['CLMINSUR'] = pd.DataFrame(mode_imputer.fit_transform(claim_df[['CLMINSUR']]))
claim_df['SEATBELT'] = pd.DataFrame(mode_imputer.fit_transform(claim_df[['SEATBELT']]))

claim_df.CLMAGE.mean() #28.41442
claim_df.CLMAGE.median() #30.0

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')

claim_df['CLMAGE'] = pd.DataFrame(median_imputer.fit_transform(claim_df[['CLMAGE']]))

#check for NaN
claim_df.isna().sum()


#Identify duplicate records in the data
duplicate = claim_df.duplicated()  
duplicate

sum(duplicate)

# visualize the data 
claim_df.plot(kind='box', subplots=True, sharey= False, figsize = (20,7))
plt.subplots_adjust(wspace=0.5)
plt.show()

claim_df.describe()
