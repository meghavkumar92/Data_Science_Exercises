# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:14:51 2022

@author: Megha
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_inf = pd.read_csv(r'D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/Assignment_module02 (1).csv')
data_inf.info()

#Q5

data_inf.Points.mean()
data_inf.Points.median()
data_inf.Points.mode()

data_inf.Score.mean()
data_inf.Score.median()
data_inf.Score.mode()

data_inf.Weigh.mean()
data_inf.Weigh.median()
data_inf.Weigh.mode()

#OR
data_inf.mean()
data_inf.median()
data_inf.mode()
                
#
data_inf.var()
data_inf.std()
    

range_p = max(data_inf.Points) - min(data_inf.Points)
range_p

range_s = max(data_inf.Score) - min(data_inf.Score)
range_s

range_w = max(data_inf.Weigh) - min(data_inf.Weigh)
range_w

data_inf.describe()

#visualize the data 
data_inf.plot(kind='box', subplots=True, sharey= False, figsize = (32,3))
plt.subplots_adjust(wspace=0.75)
plt.show()


#Q7

comp_data = [['Allied Signal', 24.23], ['Bankers Trust', 25.53], ['General Mills', 25.41], ['ITT Industries', 24.14], ['J.P.Morgan & Co.',29.62], ['Lehman Brothers',28.25], ['Marriott',25.81], ['MCI', 24.39], ['Merrill Lynch', 40.26], ['Microsoft', 32.95], ['Morgan Stanley', 91.36], ['Sun Microsystems', 25.99], ['Travelers', 39.42], ['US Airways', 26.71], ['Warner-Lambert', 35.00]]

comp_df = pd.DataFrame(comp_data, columns=['Name of company','Measure X'])
comp_df.info()

comp_df['Measure X'].mean()
comp_df['Measure X'].median()


comp_df['Measure X'].var()
comp_df['Measure X'].std() 

comp_df['Measure X'].skew()
comp_df['Measure X'].kurt()

sns.distplot(comp_df['Measure X'], bins=np.linspace(20,150,10))

plt.hist(comp_df['Measure X'], bins=6, color='blue', edgecolor='yellow')

#scatter plot
plt.scatter(x = comp_df['Measure X'], y = comp_df['Name of company']) 

#visualize the data 
comp_df.plot(kind='box', subplots=True, sharey= False, figsize = (15,2))
plt.subplots_adjust(wspace=0.75)
plt.show()
