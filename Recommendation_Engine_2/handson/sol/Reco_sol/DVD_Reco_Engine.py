# import all the required libraries
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sqlalchemy import create_engine
import mysql.connector as connector

#import dataset
game_dvd = pd.read_csv('D:/360digi/DS/Sharath/Recommendation_Engine_2/handson/Datasets_Recommendation Engine/game.csv')
game_dvd.shape #(5000, 3)

game_dvd.info()

#Database Connection

#Upload the Table into Database

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user = "newuser", pw = "newUser999", db = "assignment"))

game_dvd.to_sql('game', con = engine, if_exists= 'append', chunksize= 1000, index= False)



#Read the game table from MySQL database

con = connector.Connect(host = 'localhost', port = '3306', user = 'newuser', password = 'newUser999', database = 'assignment', auth_plugin = 'mysql_native_password')

cur = con.cursor()
con.commit()

cur.execute('SELECT * FROM game')
df = cur.fetchall()

game_df = pd.DataFrame(df)
game_df = game_df.rename({0 : 'userId'}, axis = 1)
game_df = game_df.rename({1 : 'game'}, axis = 1)
game_df = game_df.rename({2 : 'rating'}, axis = 1)

game_df.info()  #(5000, 3)


# Check for missing values
game_df.isnull().sum()
# or
game_df['game'].isnull().sum()

# no missing values

#Create a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")

# Transform a count matrix to a normalized tf-idf representation
tfidf_matrix = tfidf.fit(game_df.game)

#Save the pipeline for tfidf matrix
joblib.dump(tfidf_matrix, 'game_matrix')

os.getcwd()

mat = joblib.load('game_matrix')

tfidf_matrix = mat.transform(game_df.game)

tfidf_matrix.shape # (5000, 3068)

# Cosine similarity
# cosine(x, y)= (x.y⊺) / (||x||.||y||)

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

joblib.dump(cosine_sim_matrix, 'cosine_matrix')

# create a mapping of game name to index number
game_index = pd.Series(game_df.index, index = game_df['game']).drop_duplicates()


#Example

# Super Mario Galaxy 2  #6
game_id1 = game_index['Super Mario Galaxy 2']
game_id1


# Nintendo Land #4853
game_id3 = game_index['Perfect Dark']
game_id3 

# Custom Function to Find the TopN games to be Recommended
from collections.abc import Iterable

def get_recommendations(Name, topN):
    # Get the game index using its title
    game_id = game_index[Name]
    
    if isinstance(game_id,Iterable):
        game_id = game_id[0]
    
    #print(game_id)
    # Getting the pair wise similarity score for all the games with the requested game 
    # (extract all columns values in cosine similarity matrix for the game id. ) 
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # sort the cosine_similarity scores based on score
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar games
    cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Get the game index
    game_idx = [i[0] for i in cosine_scores_N]
    game_scores = [i[1] for i in cosine_scores_N]
    
    #similar games and its scores
    game_similar_show = pd.DataFrame(columns = ["name", "Score"])
    game_similar_show["name"] = game_df.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)
    return(game_similar_show.iloc[1:, ])

rec = get_recommendations("Perfect Dark", topN = 10) # Two rows have same game name 

rec

res = get_recommendations("Nintendo Land", topN = 5)

res

get_recommendations("Super Mario Galaxy 2", topN = 10)
# Angry Birds 4917
