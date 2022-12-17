import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDF(documents):
    '''
    inputs:
        documents: this is a list of documents where each document is a string of words
    outputs:
        features: this is a list of features of each document, so features[0] is the features of doc1
                  which is the tfidf of this document  for each word in the vocabulary.
                  This is of N lists where N is the number of documnets 
                  and V elements in each of the N lists where V is the vocab size  
    '''
    vectorizer =TfidfVectorizer()
    vectors= vectorizer.fit_transform(documents)
    dense=vectors.todense()
    # change from 2d array (dense) tp 1d array 
    features=dense.tolist()
    return features

# doca ='انا طالبه  في هندسة'
# docb= 'انا سعيدة جدا'
# print(TFIDF([doca,docb]))