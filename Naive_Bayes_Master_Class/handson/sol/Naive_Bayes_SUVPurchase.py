import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as skmet


car_df = pd.read_csv("D:/360digi/DS/Sharath/Naive_Bayes_Master_Class/handson/Datasets_Naive Bayes/NB_Car_Ad.csv")
car_df.info()

car_df.drop(['User ID'], axis = 1, inplace = True)

car_df.describe()

car_df['target'] = np.where(car_df['Purchased'] == 0, 'Not_Purchased', car_df['Purchased'] )
car_df['target'] = np.where(car_df['target'] == '1', 'Purchased', car_df['target'] )

car_df.isnull().sum() # No missing values

# Preprocessing dataset

categorical_features = ['Gender']
categorical_features


categ_pipeline = Pipeline([('label', DataFrameMapper([(categorical_features, OneHotEncoder() )]) )])

preprocess_pipeline = ColumnTransformer([( 'categorical', categ_pipeline, categorical_features) ])

processed = preprocess_pipeline.fit(car_df)

car_data = pd.DataFrame(processed.transform(car_df))
car_data

a = ['Purchased']
numerical_cols = list(filter(lambda x: x != a[0] ,car_df.select_dtypes(include = ['int']).columns))
numerical_cols

car_data = pd.concat([car_data, car_df[numerical_cols]], axis = 1)
car_data.columns

car_data.rename(columns = {0:'Female', 1:'Male'}, inplace = True)


numerical_features = car_data.select_dtypes(exclude = ['object']).columns
numerical_features


scale_pipeline = Pipeline([('scale', MinMaxScaler())])

preprocess_pipeline2 = ColumnTransformer([('scale', scale_pipeline, numerical_features)], remainder= 'passthrough')

processed2 = preprocess_pipeline2.fit(car_data)

car_norm = pd.DataFrame(processed2.transform(car_data))

car_norm.describe()


# Separating the input and output from the dataset
X = np.array(car_norm.iloc[:,:])
Y = np.array(car_df['Purchased'])

X
Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train.shape

X_test.shape


car_df.target.value_counts()
# Not_Purchased    257
# Purchased        143

# ***Multinomial Naive Bayes***

classifier_mb = MultinomialNB()
classifier_mb.fit(X_train, Y_train)

#Evaluation on test data
test_pred_m = classifier_mb.predict(X_test)
pd.crosstab(Y_test, test_pred_m)
skmet.accuracy_score(Y_test, test_pred_m)  # 0.725

#Training Data accuracy
train_pred_m = classifier_mb.predict(X_train)
pd.crosstab(Y_train, train_pred_m)
skmet.accuracy_score(Y_train, train_pred_m)  # 0.621875

# Model Tuning - Hyperparameter optimization
mnb_lap = MultinomialNB(alpha = 5)
mnb_lap.fit(X_train, Y_train)

#Evaluation on Test data
test_pred_lap = mnb_lap.predict(X_test)
pd.crosstab(Y_test, test_pred_lap)
skmet.accuracy_score(Y_test, test_pred_lap) # 0.725

# Training Data accuracy
train_pred_lap = mnb_lap.predict(X_train)
pd.crosstab(train_pred_lap, Y_train)
skmet.accuracy_score(Y_train, train_pred_lap) # 0.621875

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y_train, train_pred_m)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not_purchased', 'Purchased'])
cmplot.plot()
cmplot.ax_.set(title = 'Luxury SUV car purchase Classification Confusion Matrix (Train)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y_test, test_pred_m)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not_purchased', 'Purchased'])
cmplot.plot()
cmplot.ax_.set(title = 'Luxury SUV car purchase Classification Confusion Matrix (Test)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')



# 121 rows of 'Purchased' data in X_train was getting predicted as 'Not Purchased'. 
# The data was imbalanced and the accuracy of the model was nearly 62%
# In order to balance the data we consider a small difference of random delta value and train the model.
# We can see improvement in model accuracy by 72%.

#################### Sampling the Training data with purchased rows to improve the imbalance


X_pur_dp,Y_pur_dp = X_train[(Y_train == 1)].copy()[:78], Y_train[(Y_train == 1)].copy()[:78]

delta2 = .01*np.random.rand(78,2)
delta1 = np.random.randint(-1,2, (78,2))

X_pur_dp = X_pur_dp + np.hstack([delta1,delta2])

X_pur_dp[:,:2] = np.abs(X_pur_dp[:,:2])%2


X_train_ex, Y_train_ex = np.vstack([X_train,X_pur_dp]), np.hstack([Y_train,Y_pur_dp])


plt.hist(Y_train_ex)
plt.hist(Y_train_ex);plt.hist(Y_train)



classifier_pur = MultinomialNB()
classifier_pur.fit(X_train_ex, Y_train_ex)

#Training Data accuracy
train_pred_pur = classifier_pur.predict(X_train_ex)
pd.crosstab(Y_train_ex, train_pred_pur)
skmet.accuracy_score(Y_train_ex, train_pred_pur)  # 0.621875 # without delta =  0.7311 # with delta = 0.7286


#Evaluation on test data
test_pred_pur = classifier_pur.predict(X_test)
pd.crosstab(Y_test, test_pred_pur)
skmet.accuracy_score(Y_test, test_pred_pur) # 0.725 # without delta = 0.7875 # with delta = 0.8125

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y_train_ex, train_pred_pur)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not_purchased', 'Purchased'])
cmplot.plot()
cmplot.ax_.set(title = 'Luxury SUV car purchase Classification Confusion Matrix (Train with delta) ', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(Y_test, test_pred_pur)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Not_purchased', 'Purchased'])
cmplot.plot()
cmplot.ax_.set(title = 'Luxury SUV car purchase Classification Confusion Matrix (Test with delta)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')