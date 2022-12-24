import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec


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

def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in word2vec_model.wv.index_to_key]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return []

def CBOW(train_documents,test_documents):
    '''
    inputs:
        documents: this is a list of documents where each document is a a list of preprcessed tokens
    outputs:
        features:   This is a 2D array  with N rows and V cols
                    where N is the number of words &  V is the vector size
                    this is a list of features of each word in each document, so features[0] is the vector of word[0]
    '''
    # Create CBOW (Continuous Bag of Words) model
    # vector_size should be tuned according to the size of the vocabulary we have, could be 100 for the normal dataset
    vectorizer_train = Word2Vec(train_documents, min_count = 1, vector_size = 300, window = 5)
    vectorizer_test = Word2Vec(test_documents, min_count = 1, vector_size = 300, window = 5)
    # append the vectors of all sentences to a list
    vectors_cbow_train = []
    # get the vector of all sentences 
    for sentence in train_documents:
        vec = get_mean_vector(vectorizer_train, sentence)
        if len(vec) > 0:
            vectors_cbow_train.append(vec)

    vectors_cbow_test = []
    for sentence in test_documents:
        vec = get_mean_vector(vectorizer_test, sentence)
        if len(vec) > 0:
            vectors_cbow_test.append(vec)

    return np.array(vectors_cbow_train), np.array(vectors_cbow_test)
    # return vectorizer_train.wv[vectorizer_train.wv.index_to_key], vectorizer_test.wv[vectorizer_test.wv.index_to_key]


def SG(train_documents,test_documents):
    '''
    inputs:
        documents: this is a list of documents where each document is a a list of preprcessed tokens
    outputs:
        features:   This is a 2D array  with N rows and V cols
                    where N is the number of words &  V is the vector size
                    this is a list of features of each word in each document, so features[0] is the vector of word[0]
    '''
    # create skip-gram model (SG) 
    # vector_size should be tuned according to the size of the vocabulary we have, could be 100 for the normal dataset
    vectorizer_train = Word2Vec(train_documents, min_count = 1, vector_size = 300, window = 5, sg=1)
    vectorizer_test = Word2Vec(test_documents, min_count = 1, vector_size = 300, window = 5, sg=1)
    # append the vectors of all sentences to a list
    vectors_sg_train = []
    # get the vector of all sentences 
    for sentence in train_documents:
        vec = get_mean_vector(vectorizer_train, sentence)
        if len(vec) > 0:
            vectors_sg_train.append(vec)

    vectors_sg_test = []
    for sentence in test_documents:
        vec = get_mean_vector(vectorizer_test, sentence)
        if len(vec) > 0:
            vectors_sg_test.append(vec)

    return np.array(vectors_sg_train), np.array(vectors_sg_test)


# def applyPCA(X,n_components=100):
#     X_copy=X.copy()
#     pca=PCA(n_components=n_components)
#     X_copy=pca.fit_transform(X_copy)
#     return X_copy


# Just a demmo test
doca ='انا طالبه  في هندسة'.split()
docb= 'انا سعيدة جدا'.split()
docc='انا هاله'.split()

# print(SG( [doca,docb],[docc]))
# print(CBOW( [doca,docb],[docc]))
# print(TFIDF([doca,docb],[docc]))
# print(BOW([doca,docb],[docc]))

