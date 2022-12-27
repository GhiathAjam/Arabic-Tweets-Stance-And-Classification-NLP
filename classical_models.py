from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

def SVMmodel(Xtrain,y_train,X_test,kernel='rbf'):
    clf=svm.SVC(C=10, class_weight='balanced', kernel=kernel)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def LSVMmodel(Xtrain,y_train,X_test):
    clf=svm.LinearSVC(class_weight='balanced')
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def NBmodel(Xtrain,y_train,X_test):
    clf=GaussianNB()
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def MNBmodel(Xtrain,y_train,X_test):
    clf=MultinomialNB()
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def KNNmodel(Xtrain,y_train,X_test,k=5):
    clf=KNeighborsClassifier(n_neighbors=k)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def DTmodel(Xtrain,y_train,X_test):
    clf=DecisionTreeClassifier(class_weight='balanced')
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def RFmodel(Xtrain,y_train,X_test):
    clf=RandomForestClassifier(n_estimators=1000, class_weight='balanced')
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def LRmodel(Xtrain,y_train,X_test, class_weight='balanced'):
    clf=LogisticRegression(max_iter=300, class_weight=class_weight)
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

def RGmodel(Xtrain,y_train,X_test):
    clf=RidgeClassifier(class_weight='balanced')
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred

