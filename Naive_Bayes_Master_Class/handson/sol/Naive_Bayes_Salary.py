# import all the required libraries and modules
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as skmet


salary_df = pd.read_csv("D:/360digi/DS/Sharath/Naive_Bayes_Master_Class/handson/Datasets_Naive Bayes/SalaryData_Train.csv")
salary_df.info()

salary_df.head(5)

salary_df['target'] = np.where(salary_df['Salary'] == ' <=50K', 'Below', salary_df['Salary'] )
salary_df['target'] = np.where(salary_df['target'] == ' >50K', 'Above', salary_df['target'] )

salary_df.describe()

salary_df.isnull().sum() # No missing values

numeric_features = salary_df.select_dtypes(exclude = ['object']).columns

numeric_features

# Preprocessing the train dataset

a = ['Salary', 'target']
cat_cols = list(filter(lambda x: x != a[0] and x != a[1],salary_df.select_dtypes(exclude = ['int']).columns))

categorical_features = cat_cols
categorical_features
categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features, OneHotEncoder(drop='if_binary') )]) )])

preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features)], remainder = 'passthrough')

processed = preprocess_pipeline.fit(salary_df)

salary_new = pd.DataFrame(processed.transform(salary_df))
salary_new.info()

# Normalized data frame (considering the numerical part of data)

new_features = salary_new.select_dtypes(exclude = ['object']).columns
new_features

scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, new_features)], remainder = 'passthrough')

processed2 = preprocess_pipeline2.fit(salary_new.iloc[:,:101])

processed2

salary_norm = pd.DataFrame(processed2.transform(salary_new.iloc[:,:101]))
salary_norm.describe()

# labeling target values 
salary_df['Y'] = np.where(salary_df.target == 'Above', 1, 0)

#concatenate two dataframe 
salary = pd.concat([salary_norm, salary_df['Y']], axis = 1)
salary.columns


# Separating the input and output from the dataset
X = np.array(salary_norm.iloc[:, :])
Y = np.array(salary_df['Y'])
X



#+++++++++++++++++++++++++++++++Test Dataset ++++++++++++++++++++++++++++++++++++++++++++++++++++

Test_df = pd.read_csv("D:/360digi/DS/Sharath/Naive_Bayes_Master_Class/handson/Datasets_Naive Bayes/SalaryData_Test.csv")
Test_df

Test_df['target'] = np.where(Test_df['Salary'] == ' <=50K', 'Below', Test_df['Salary'] )
Test_df['target'] = np.where(Test_df['target'] == ' >50K', 'Above', Test_df['target'] )

Test_df.describe()

Test_df.isnull().sum() # No missing values

numeric_feat = Test_df.select_dtypes(exclude = ['object']).columns

numeric_feat


#####Preprocessing the test dataset ##############

b = ['Salary', 'target']
cate_cols = list(filter(lambda x: x != b[0] and x != b[1],Test_df.select_dtypes(exclude = ['int']).columns))

cate_cols

categ_pipeline1 = Pipeline([('label', DataFrameMapper([(cate_cols, OneHotEncoder(drop='if_binary') )]) )])

preprocess_pipeline1 = ColumnTransformer([('categorical', categ_pipeline1, cate_cols)], remainder = 'passthrough')


processed1 = preprocess_pipeline1.fit(Test_df)


Test_df_new = pd.DataFrame(processed1.transform(Test_df))
Test_df_new.info()

#####Normalization of test data ##################################################
new_feat = Test_df_new.select_dtypes(exclude = ['object']).columns
new_feat

preprocess_pipeline3 = ColumnTransformer([('scale', scale_pipeline, new_feat)], remainder = 'passthrough')

processed3 = preprocess_pipeline3.fit(Test_df_new.iloc[:,:101])
processed3


test_norm = pd.DataFrame(processed3.transform(Test_df_new.iloc[:,:101]))
test_norm.describe()
##
Test_df['Y'] = np.where(Test_df.target == 'Above', 1, 0)

test = pd.concat([test_norm, Test_df['Y']], axis = 1)
test.columns
#####Split the data set ##########################
X_test = np.array(test_norm.iloc[:, :])
Y_test = np.array(Test_df['Y'])
X_test



# ***Multinomial Naive Bayes***

classifier_mb = MultinomialNB()
classifier_mb.fit(X, Y)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(X_test)

pd.crosstab(Y_test,test_pred_m )

skmet.accuracy_score(Y_test, test_pred_m) # 0.7749


# Training Data accuracy
train_pred_m = classifier_mb.predict(X)

pd.crosstab(Y, train_pred_m)

skmet.accuracy_score(Y, train_pred_m) # 0.7729


# Model Tuning - Hyperparameter optimization

mnb_lap = MultinomialNB(alpha = 5)
mnb_lap.fit(X, Y)

# Evaluation on Test Data
test_pred_lap = mnb_lap.predict(X_test)

pd.crosstab(Y_test,test_pred_lap )

skmet.accuracy_score(Y_test, test_pred_lap)  # 0.7749


# Training Data accuracy
train_pred_lap = mnb_lap.predict(X)

pd.crosstab(Y, train_pred_lap)

skmet.accuracy_score(Y, train_pred_lap) # 0.7729



# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y, train_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [' <=50K', ' >50K'])
cmplot.plot()
cmplot.ax_.set(title = 'Salary Classification Confusion Matrix(Train)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y_test, test_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [' <=50K', ' >50K'])
cmplot.plot()
cmplot.ax_.set(title = 'Salary Classification Confusion Matrix(Test)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')











# To find the number of unique categories in each columns #

cols = salary_df.select_dtypes(include = ['object']).columns

cnt = 0
for column in cols:
    print(column)
    cnt += len(salary_df[column].unique())
    print(cnt)


res = salary_new.describe()

np.sum(Y) # 7508 for train 1 - Above 50K

Y.size - np.sum(Y) #  22653 for train 0 - Below 50K


np.sum(Y_test) # 3700 for test 1 
Y_test.size - np.sum(Y_test) # 11360 for test 0
