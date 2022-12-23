from sklearn import svm


def SVMmodel(Xtrain,y_train,X_test,kernel='rbf'):
    clf=svm.SVC(kernel=kernel,decision_function_shape='ovr')
    clf.fit(X=Xtrain,y=y_train)
    y_pred=clf.predict(X=X_test)
    return y_pred