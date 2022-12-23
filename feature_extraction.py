import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def TFIDF(documents):
    '''
    inputs:
        documents: this is a list of documents where each document is a list of preprcessed tokens
    outputs:
        features: This is a 2D array  with N rows and V cols
                  where N is the number of documnets &  V is the vocab size  
                  this is a list of features of each document, so features[0] is the features of doc1
                  which is the tfidf of this document  for each word in the vocabulary.
    '''
    vectorizer =TfidfVectorizer(analyzer=lambda x: x)
    vectors= vectorizer.fit_transform(documents)
    features=vectors.todense()
    return features


def BOW(documents):
    '''
    inputs:
        documents: this is a list of documents where each document is a a list of preprcessed tokens
    outputs:
        features: This is a 2D array  with N rows and V cols
                  where N is the number of documnets &  V is the vocab size  
                  this is a list of features of each document, so features[0] is the features of doc1
                  which is the count of each word in  this document  for each word in the vocabulary.            
    '''
    # count vectorizer function takes sentences as input 
    # convert it into matrix representation 
    # where each cell will be filled by the frequency of each vocab
    vectorizer = CountVectorizer(analyzer=lambda x: x)
    bow_model = vectorizer.fit_transform(documents)
    return bow_model.toarray()


# Just a demmo test
doca ='انا طالبه  في هندسة'.split()
docb= 'انا سعيدة جدا'.split()
print(TFIDF([doca,docb]))
print(BOW([doca,docb]))