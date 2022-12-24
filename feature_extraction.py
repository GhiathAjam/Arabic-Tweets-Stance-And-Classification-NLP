import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def TFIDF(train_documents,test_documents):
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
    train_features= vectorizer.fit_transform(train_documents)
    test_features=vectorizer.transform(raw_documents=test_documents)
    return train_features.toarray(), test_features.toarray()
    


def BOW(train_documents,test_documents):
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
    train_features = vectorizer.fit_transform(train_documents)
    test_features=vectorizer.transform(raw_documents=test_documents)
    return train_features.toarray(), test_features.toarray()



# def applyPCA(X,n_components=100):
#     X_copy=X.copy()
#     pca=PCA(n_components=n_components)
#     X_copy=pca.fit_transform(X_copy)
#     return X_copy


# Just a demmo test
doca ='انا طالبه  في هندسة'.split()
docb= 'انا سعيدة جدا'.split()
docc='انا هاله'.split()

# print(TFIDF([doca,docb],[docc]))
# print(BOW([doca,docb],[docc]))
