import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



def Normalize_train_test_features(train_features, test_features):

    scaler = preprocessing.StandardScaler()

    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features




def Load_label_toarray(label_data):

    label_data = label_data.as_matrix(columns=None)
    label_data = label_data.ravel()

    return label_data


def Load_features_toarray(features_data):

    features_data  = features_data.as_matrix(columns=None)

    return features_data


def Univariate_selection(train_features, train_label, n_features):

    new_features = SelectKBest(chi2, k=n_features).fit_transform(train_features, train_label)

    print new_features.shape


def Tree_based_selection(train_features, train_label):

    print "Before feature selection, the raw feature space has %d dimension with %d training sample " % (len(train_features[0]), len(train_features))
    clf = ExtraTreesClassifier()
    clf = clf.fit(train_features, train_label)
    model = SelectFromModel(clf, prefit=True)
    features_new = model.transform(train_features)

    print "After feature selection, the new feature space has %d dimension with %d training sample " % (
    len(features_new[0]), len(features_new))

    pass


def L1_selection(train_features, train_label):

    print "Before feature selection, the raw feature space has %d dimension with %d training sample " % (len(train_features[0]), len(train_features))
    clf =  LinearSVC(C=10, penalty="l1", dual=False)
    clf = clf.fit(train_features, train_label)
    model = SelectFromModel(clf, prefit=True)
    features_new = model.transform(train_features)

    print "After feature selection, the new feature space has %d dimension with %d training sample " % (
    len(features_new[0]), len(features_new))
    pass


def Remove_low_variance(train_features, train_label):

    print "Before feature selection, the raw feature space has %d dimension with %d training sample " % (
    len(train_features[0]), len(train_features))

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    features_new = sel.fit_transform(train_features)

    print "After feature selection, the new feature space has %d dimension with %d training sample " % (
    len(features_new[0]), len(features_new))


def Select_best_features(train_features,train_label):


    while True:

        estimators = [('reduce_dim', PCA()), ('clf', SVC())]
        pipe = Pipeline(estimators)
        param_grid = dict(reduce_dim=[  VarianceThreshold(threshold=(.8 * (1 - .8))),
                                        PCA(32),
                                        PCA(48),
                                        SelectFromModel(LinearSVC(C=1, penalty="l1", dual=False)),
                                        SelectFromModel(ExtraTreesClassifier())
                                        ],
                                clf=[
                                        RandomForestClassifier(),
                                     ]
                            )

        grid_search = GridSearchCV(pipe, param_grid=param_grid)
        grid_search = grid_search.fit(train_features, train_label)

        best_clf = grid_search.best_estimator_

        scores = cross_val_score(best_clf, train_features, train_label, cv=5)

        print "Selecting features: score %f, continue......" % np.mean(scores)


        if np.mean(scores)> 0.945 and grid_search.best_estimator_.named_steps['reduce_dim'] != None:

            print "Choose this feature selection tool: "
            print best_clf

            break

    return grid_search.best_estimator_.named_steps['reduce_dim']