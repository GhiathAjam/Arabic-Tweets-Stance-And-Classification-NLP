from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def SVMmodel(Xtrain,y_train,X_test,kernel='rbf'):
    # clf=svm.SVC(kernel=kernel,decision_function_shape='ovr')
    clf=svm.SVC(C=10)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def NBmodel(Xtrain,y_train,X_test):
    clf=GaussianNB()
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def KNNmodel(Xtrain,y_train,X_test,k=1):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def DTmodel(Xtrain,y_train,X_test):
    clf=DecisionTreeClassifier()
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def RFmodel(Xtrain,y_train,X_test):
    clf=RandomForestClassifier(n_estimators=300)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def LRmodel(Xtrain,y_train,X_test):
    clf=LogisticRegression(max_iter=200)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred
