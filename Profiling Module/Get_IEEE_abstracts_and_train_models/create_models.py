from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def train_models():

    with open('dataset/ieee_dataset_six_areas.csv', encoding="utf-8")as file:
        data = pd.read_csv(file)

    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics

    X_train, X_test, y_train, y_test = train_test_split(data.abstract, data.label, test_size=0.2)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', SVC(kernel='linear', gamma=7, C=1))])

    X_train, X_test, y_train, y_test = train_test_split(data.abstract, data.lable, test_size=0.2)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', SVC(kernel='rbf', gamma=0.5, C=100, probability=True))])

    X_train, X_test, y_train, y_test = train_test_split(data.abstract, data.lable, test_size=0.2)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', SVC(kernel='poly', degree=6, C=1))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', SVC(kernel="sigmoid", gamma=7, probability=True))])
    #
    X_train, X_test, y_train, y_test = train_test_split(data.abstract, data.lable, test_size=0.2)
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', LinearSVC(C=10))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', KNeighborsClassifier(n_neighbors=7))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf',RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', MultinomialNB())])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', BernoulliNB(alpha=7.0, binarize=0.0, class_prior=None, fit_prior=True))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=1000)),
                         ('clf', LogisticRegression(random_state=0))])

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100)),
                         ('clf', DecisionTreeClassifier(max_depth=7))])


    model = pipeline.fit(X_train, y_train)

    # vectorizer = model.named_steps['vect']
    # chi = model.named_steps['chi']
    # clf = model.named_steps['clf']
    # print(chi)
    #
    # feature_names = vectorizer.get_feature_names()
    # feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
    # feature_names = np.asarray(feature_names)
    #
    target_names = ['1', '2', '3', '4', '5', '6']
    # for i, label in enumerate(target_names):
    #     top100 = np.argsort(clf.coef_[i])[-100:]
    #     print("%s: %s" % (label, " ".join(feature_names[top100])))

    # model = pipeline.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(accuracy*100)

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    print(metrics.classification_report(y_test, predictions, target_names=target_names))
    #
    filename = 'ieee_SVC_kernal=rbf_g05_prob.sav'
    pickle.dump(pipeline, open(filename, 'wb'))


train_models()
