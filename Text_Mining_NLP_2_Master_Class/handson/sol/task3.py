# task 3
# website - https://www.nykaa.com/ 
# https://www.nykaa.com/huda-beauty-fauxfilter-luminous-matte-full-coverage-liquid-foundation/reviews/1498448?skuId=1498438&ptype=reviews


import requests
from bs4 import BeautifulSoup as bs
import re

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

product_reviews = []

for i in range(1, 21):
    ip = []
    url = "https://www.nykaa.com/huda-beauty-fauxfilter-luminous-matte-full-coverage-liquid-foundation/reviews/1498448?skuId=1498438&ptype=reviews"
    response = requests.get(url)
    soup = bs(response.content, "html.parser")
    reviews = soup.find_all("p", attrs= {"class", "css-1n0nrdk"})
    for i in range(len(reviews)):
        ip.append(reviews[i].text)
        
    product_reviews = product_reviews + ip
    

# Joining all the reviews into single paragraph 
ip_rev_string = " ".join(product_reviews)

# Removing unwanted symbols incase they exists
ip_rev_string = re.sub("[^A-Za-z" "]+", " ", ip_rev_string).lower()

# Words that are contained in the reviews
ip_reviews_words = ip_rev_string.split(" ")

ip_reviews_words = ip_reviews_words[1:]

#TFIDF
vectorizer = TfidfVectorizer(use_idf = True, ngram_range = (1, 1))
X = vectorizer.fit_transform(ip_reviews_words)

with open("D:/360digi/DS/Sharath/Text_Mining_NLP_2_Master_Class/handson/sol/stop.txt", "r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]


# Joining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)



# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(background_color= 'White', width= 1800, height= 1400).generate(ip_rev_string)
plt.imshow(wordcloud_ip)

# Positive words # Choose the path for +ve words stored in system
with open("D:/360digi/DS/Sharath/Text_Mining_NLP_2_Master_Class/handson/sol/positive-words.txt", "r") as pos:
  poswords = pos.read().split("\n")


# Positive word cloud
# Choosing only words which are present in positive words

ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])


wordcloud_pos_in_pos = WordCloud(background_color = 'White', width = 1800,  height = 1400).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)


# Negative word cloud
# Choose path for -ve words stored in system
with open("D:/360digi/DS/Sharath/Text_Mining_NLP_2_Master_Class/handson/sol/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")


# negative word cloud
# Choosing only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])


wordcloud_neg_in_neg = WordCloud(background_color = 'black', width = 1800, height = 1400).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)


#####################Sentiment analysis
ip_rev_string = " ".join(product_reviews)


# Wordcloud with bigram
nltk.download('punkt')

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = ip_rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars as well as stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['feel', 'wow', 'mascara'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

# nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
vectorizer = CountVectorizer(ngram_range = (2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width, stopwords = new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()







