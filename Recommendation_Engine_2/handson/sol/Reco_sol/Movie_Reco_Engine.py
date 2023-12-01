# import all the required libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sqlalchemy import create_engine
import mysql.connector as connector

#import dataset
movies = pd.read_csv("D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Recommendation Engine/Entertainment.csv")
movies.info()

movies.shape #(51, 4)

#Database Connection

#Upload the Table into Database

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "newuser", pw = "newUser999", db = "assignment"))

movies.to_sql('movies', con = engine, if_exists = 'append', chunksize= 1000, index = False)

#Read the game table from MySQL database

con = connector.Connect(host = 'localhost', port = '3306', user = 'newuser', password = 'newUser999', database = 'assignment', auth_plugin = 'mysql_native_password')

cur = con.cursor()
con.commit()

cur.execute('SELECT * FROM movies')
df = cur.fetchall()

movies_df = pd.DataFrame(df)
movies_df = movies_df.rename({0 : 'movie_id'}, axis = 1)
movies_df = movies_df.rename({1 : 'title'}, axis = 1)
movies_df = movies_df.rename({2 : 'category'}, axis = 1)
movies_df = movies_df.rename({3 : 'reviews'}, axis = 1)

# Check for missing values
movies_df.isnull().sum()

# no missing values

#Create a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")

tfidf_matrix = tfidf.fit(movies_df.category)

joblib.dump(tfidf_matrix, 'movie_matrix')

os.getcwd()

mat = joblib.load("movie_matrix")

tfidf_matrix = mat.transform(movies_df.category)

tfidf_matrix.shape #(51, 34)

# Cosine similarity
# cosine(x, y)= (x.y⊺) / (||x||.||y||)

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

joblib.dump(cosine_sim_matrix, 'movies_cosine_matrix')

movies_index = pd.Series(movies_df.index, index = movies_df['title']).drop_duplicates()

#Example
# Sudden Death (1995) #8
movie_id = movies_index["Sudden Death (1995)"]
movie_id

# To Die For (1995) #41

movie_id = movies_index["To Die For (1995)"]
movie_id



# Custom Function to Find the TopN games to be Recommended

def get_recommendations(Title, topN):
    movie_id = movies_index[Title]
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    cosine_scores_N = cosine_scores[0: topN + 1]
    movie_idx = [i[0] for i in cosine_scores_N]
    movie_scores = [i[1] for i in cosine_scores_N]
    
    # similar movies and scores
    movie_similar_show = pd.DataFrame(columns = ["title", "score"])
    movie_similar_show["title"] = movies_df.loc[movie_idx, "title"]
    movie_similar_show["score"] = movie_scores
    movie_similar_show.reset_index(inplace = True)
    return(movie_similar_show.iloc[1:, ])


rec = get_recommendations("Waiting to Exhale (1995)", topN= 5)

rec





movie_id = movies_index["Waiting to Exhale (1995)"]
movie_id


# Waiting to Exhale (1995)  #3 	# category:Sci-Fi, Thriller














