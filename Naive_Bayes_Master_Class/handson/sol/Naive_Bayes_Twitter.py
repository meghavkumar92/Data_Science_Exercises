# import all the required libraries and modules
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy import text

from sklearn.model_selection import train_test_split

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as skmet
import joblib

tweets_df = pd.read_csv("D:/360digi/DS/Sharath/Naive_Bayes_Master_Class/handson/Datasets_Naive Bayes/Disaster_tweets_NB.csv", encoding = "ISO-8859-1")

tweets_df.info()

tweets_df['spam'] = np.where(tweets_df.target == 1, 'real', 'fake')

# PostgreSQL

conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}".format(user = "postgres", pw = "postgres", db = "assignment"))

db = create_engine(conn_string)
conn = db.connect()

tweets_df.to_sql('tweets', con = db, if_exists= 'replace', index = False)

conn.autocommit = True

sql = 'SELECT * from tweets'

tweets_data = pd.read_sql_query(text(sql), db)


# Data Preprocessing - textual data

tweets_data.target.value_counts()

tweets_data.target.value_counts() / len(tweets_data.target)

#If one of the class is <30% then the dataset is imbalanced.


# Data Split
tweet_train, tweet_test = train_test_split(tweets_data, test_size = 0.2, stratify = tweets_data[['target']], random_state = 0)
#

#Convert Textual data(Unstructured) into structured 
#CountVectorizer - Convert a collection of text documents to a matrix of token counts

countvectorizer = CountVectorizer(analyzer = 'word', stop_words = 'english')

####
s_sample = tweet_train.loc[tweet_train.text.str.len() < 50].sample(3, random_state = 35)
s_sample = s_sample.iloc[:,[3,5]]

# Document Term Matrix with CountVectorizer (# returns 1D array)
s_vec = pd.DataFrame(countvectorizer.fit_transform(s_sample.values.ravel()).toarray(), columns = countvectorizer.get_feature_names_out())
s_vec # error values[:,1]
####





# creating a matrix of token counts fo rthe entire text document
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of tweet text into word count matrix format - Bag of Words
tweets_mat = CountVectorizer(analyzer= split_into_words).fit(tweets_data.text)

all_tweets_matrix = tweets_mat.transform(tweets_data.text)

train_tweets_matrix = tweets_mat.transform(tweet_train.text)

test_tweets_matrix = tweets_mat.transform(tweet_test.text)


#SMOTE 
smote = SMOTE(random_state = 0)
X_train, Y_train = smote.fit_resample(train_tweets_matrix, tweet_train.target)


Y_train.unique()
Y_train.values.sum()
Y_train.size - Y_train.values.sum()
# The data is now balanced

## Multinomial Naive Bayes
classifier_mb = MultinomialNB() 
classifier_mb.fit(X_train, Y_train)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tweets_matrix)

pd.crosstab(tweet_test.target, test_pred_m)
#

# Accuracy
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m
# or
skmet.accuracy_score(tweet_test.target, test_pred_m)  # 0.7787


# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tweets_matrix)

pd.crosstab(tweet_train.target, train_pred_m)
#
skmet.accuracy_score(tweet_train.target, train_pred_m) # 0.9390
#

# Model Tuning - Hyperparameter optimization using laplace value
mnb_lap = MultinomialNB(alpha= 3)

mnb_lap.fit(X_train, Y_train)

# Evaluate test data after applying laplace
test_pred_lap = mnb_lap.predict(test_tweets_matrix)

pd.crosstab(test_pred_lap, tweet_test.spam) 

skmet.accuracy_score(tweet_test.target, test_pred_lap) # 0.7787
#

train_pred_lap = mnb_lap.predict(train_tweets_matrix)

pd.crosstab(train_pred_lap, tweet_train.target)
#
skmet.accuracy_score(tweet_train.target, train_pred_lap)  # 0.9390
#

# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(tweet_test.target.ravel(), test_pred_lap),
  skmet.recall_score(tweet_test.target.ravel(), test_pred_lap),
  skmet.recall_score(tweet_test.target.ravel(), test_pred_lap, pos_label = 0),
  skmet.precision_score(tweet_test.target.ravel(), test_pred_lap)))

#
# accuracy: 0.77, sensitivity: 0.67, specificity: 0.84, precision: 0.76

# Confusion Matrix - Heat Map (Test data)
cm = skmet.confusion_matrix(tweet_test.target, test_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm)#, display_labels = ['fake tweet', 'real tweet'])
cmplot.plot()
cmplot.ax_.set(title = 'Tweet Detection Confusion Matrix(Test)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')

# Confusion Matrix - Heat Map (Train data)
cm = skmet.confusion_matrix(tweet_train.target, train_pred_lap)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['fake tweet', 'real tweet'])
cmplot.plot()
cmplot.ax_.set(title = 'Tweet Detection Confusion Matrix(Train)', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


# Saving the Best Model using Pipelines
# Build the Pipeline preparing a naive bayes model on training data set
nb = MultinomialNB(alpha = 5)

pipe1 = make_pipeline(countvectorizer, smote, nb)

processed = pipe1.fit(tweet_train.text.ravel(), tweet_train.target.ravel())

joblib.dump(processed, 'NV_model')

model = joblib.load('NV_model')

test_pred = model.predict(tweet_test.text.ravel())

pd.crosstab(tweet_test.target, test_pred)

skmet.accuracy_score(tweet_test.target, test_pred)  # 0.7852


# Metrics
print("accuracy: %.2f, sensitivity: %.2f, specificity: %.2f, precision: %.2f"  %
  (skmet.accuracy_score(tweet_test.target.ravel(), test_pred),
  skmet.recall_score(tweet_test.target.ravel(), test_pred),
  skmet.recall_score(tweet_test.target.ravel(), test_pred, pos_label = 0),
  skmet.precision_score(tweet_test.target.ravel(), test_pred)))

#accuracy: 0.79, sensitivity: 0.74, specificity: 0.82, precision: 0.75

# Confusion Matrix - Heat Map
cm = skmet.confusion_matrix(tweet_test.target, test_pred)
cmplot = skmet.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['fake tweet', 'real tweet'])
cmplot.plot()
cmplot.ax_.set(title = 'Tweet Detection Confusion Matrix', 
               xlabel = 'Predicted Value', ylabel = 'Actual Value')


import os
os.getcwd()


