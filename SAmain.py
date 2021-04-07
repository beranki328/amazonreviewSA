# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:39:16 2021

@author: anjan
"""

#using Amazon Reviews as dataset of sent. analysis
#going to use Logistic Regression to train the model

import numpy as np
import pandas as pd
import sklearn
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re


#Reading the dataset and selecting the customer reviews 
df = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
X = (df['reviews.text'])
#print(X)

sentiment = {1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1}

df["sentiment"] = df["reviews.rating"].map(sentiment)
#print("keyy" + "\n")
#print(df["sentiment"])

#print(df[df["sentiment"].isnull()])
# df["sentiment"] = pd.to_numeric(df["sentiment"], errors='coerce')                                    
# df = df.dropna(subset=["sentiment"]) # removes rows/columns w/ more than one empty space 
# df["sentiment"]  = df["sentiment"] .astype(int)

#
df['reviews.text'] = df['reviews.text'].apply(lambda elem: re.sub("^[a-zA-z]", " ", str(elem)))
#follow this documentation: re.sub(pattern, repl, string, count=0, flags=0)
df['reviews.text'] = df['reviews.text'].str.lower()
wordD = df['reviews.text'].str.split()
print("debug check 1")

stopwords = stopwords.words('english')
#print(stopwords)
stemS = SnowballStemmer('english')#creating an instance of the object
df['reviews.text'] = df['reviews.text'].apply(lambda elem: [word for word in elem if not word in stopwords])
df['reviews.text'] = df['reviews.text'].apply(lambda elem: [stemS.stem(word) for word in elem])

df['cleaned'] = wordD.apply(lambda elem: ' '.join(elem))

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words=['virginamerica','unit'])
# The Count Vectorizer function converts a list of words into bag of words
# the stop words are included, and it only allows 3000 features, so basically
# the 3k most importan words.
txt=cv.fit_transform(df['cleaned']).toarray()
#print(n[0])
y=df['sentiment'].values

'''
from sklearn.feature_extraction.text import TfidfVectorizer
 
vct = TfidfVectorizer()
txt = vct.fit_transform(df['cleaned']).toarray()
# print(txt)
# txts = pd.DataFrame(txt)
# print(txts.shape)
y = df['sentiment'].values
print(y.shape)
'''


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(txt, y, test_size = 0.20, random_state = 42)

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
'''
print(X_train.shape)
print(X_test.shape)
print("Reshaping")
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
print(X_train.shape)
print(X_test.shape)
'''
logreg.fit(X_train, y_train)

from sklearn.metrics import classification_report,confusion_matrix

print('Train accuracy :', (logreg.score(X_train, y_train))*100)
print('Test accuracy :', (logreg.score(X_test, y_test))*100)
      
print('\n CONFUSION MATRIX')
print(confusion_matrix(y_train, logreg.predict(X_train)))
print('\nCLASSIFICATION REPORT')
print(classification_report(y_test, logreg.predict(X_test)))


"""
Xp = X.str.lower()
Xp = [sub.split() for sub in X]

n = []
stop_words = set(stopwords.words("english"))

for w in Xp:
    if w not in stop_words:
        n.append(w)

sf = SnowballStemmer('english')
for i in n:
    sf.stem(i)
"""