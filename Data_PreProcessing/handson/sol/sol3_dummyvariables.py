# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:51:59 2022

@author: Megha
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


ani_data = pd.read_csv('D:/360digi/DS/Sharath/Data_PreProcessing/handson/DataSets-Data Pre Processing/DataSets/animal_category.csv')
ani_data.info()

# Remove column name 'Index'
ani_data.drop(['Index'], axis=1, inplace = True)

# Check for count of NA's in each column
ani_data.isna().sum()

#new dataframe
animal_df = ani_data

# Label Encoder
# Creating instance of labelencoder
labelencoder = LabelEncoder()

ani_data['Animals'] = labelencoder.fit_transform(ani_data['Animals'])
ani_data['Gender'] = labelencoder.fit_transform(ani_data['Gender'])
ani_data['Homly'] = labelencoder.fit_transform(ani_data['Homly'])
ani_data['Types'] = labelencoder.fit_transform(ani_data['Types'])

ani_data.info()


#alternate approach
#create dummy variables
new_anidata = pd.get_dummies(animal_df, drop_first= True)



#No column labels
#One Hot Encoder
# Creating instance of One-Hot Encoder
enc = OneHotEncoder() # initializing method

enc_anidf = pd.DataFrame(enc.fit_transform(ani_data.iloc[:, 1:]).toarray())

