import pandas as pd
import os
from PreprocessingStage import *
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict



def NormalModel(train_features, train_label, test_features):
    # model_svm = SVC(kernel='rbf', C=1, degree=2)
    model_svm = SVC(kernel='linear', C=1)
    model_svm.fit(train_features, train_label)

    svm_tst_prediction = model_svm.predict(test_features)

    normal_score = cross_val_score(model_svm, train_features, train_label, cv=5)

    print svm_tst_prediction
    print normal_score
    print np.mean(normal_score)

    return svm_tst_prediction


def Get_test_features(test_features, train_features, train_label, from_file="", output_file=""):
    print "Processing test features.........."

    if from_file != "":
        ensemble_features = pd.read_csv(os.path.join(os.getcwd(), from_file))
    else:

        model_svm = SVC(kernel='rbf', C=1, degree=2)
        model_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        model_lr = LogisticRegression()
        model_lsvm = SVC(kernel='linear')
        model_rf = RandomForestClassifier()
        model_knn = KNeighborsClassifier(n_neighbors=1)
        model_dt = DecisionTreeClassifier()

        model_svm.fit(train_features, train_label)
        model_ada.fit(train_features, train_label)
        model_lr.fit(train_features, train_label)
        model_lsvm.fit(train_features, train_label)
        model_rf.fit(train_features, train_label)
        model_knn.fit(train_features, train_label)
        model_dt.fit(train_features, train_label)

        svm_tst_prediction = model_svm.predict(test_features)
        ada_tst_prediction = model_ada.predict(test_features)
        lr_tst_prediction = model_lr.predict(test_features)
        lsvm_tst_prediction = model_lsvm.predict(test_features)
        rf_tst_prediction = model_rf.predict(test_features)
        knn_tst_prediction = model_knn.predict(test_features)
        dt_tst_prediction = model_dt.predict(test_features)

        print "Get all predicted test result for seven models.........."

        predict_list = []

        for i in range(0, len(test_features)):
            prediction_value = [i, svm_tst_prediction[i], ada_tst_prediction[i],
                                lr_tst_prediction[i], \
                                lsvm_tst_prediction[i], rf_tst_prediction[i], knn_tst_prediction[i],
                                dt_tst_prediction[i]]

            predict_list.append(prediction_value)

        ensemble_features = pd.DataFrame(predict_list,
                                         columns=['test_index', 'rsvm', 'ada', 'lr', 'lsvm', 'rf', 'knn', 'dt'])

        if output_file != "":
            ensemble_features.to_csv(os.path.join(os.getcwd(), output_file), index=False)

        print ensemble_features

    return ensemble_features


def Get_ensemble_features(train_features, train_label, from_file="", output_file=''):
    print "Get all ensemble model.........."

    if from_file != "":
        ensemble_features = pd.read_csv(os.path.join(os.getcwd(), from_file))
    else:
        use_cv = 5

        kf = KFold(n_splits=use_cv, shuffle=True)
        kf.get_n_splits(train_features)
        train_test_fold = kf.split(train_features)

        prediction_value_list = []

        for train_index, test_index in train_test_fold:

            print "Get all ensemble model for one folder.........."

            tr_features, tst_features = train_features[train_index], train_features[test_index]
            tr_target, tst_target = train_label[train_index], train_label[test_index]

            model_svm = SVC(kernel='rbf', C=1, degree=2)
            model_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
            model_lr = LogisticRegression()
            model_lsvm = SVC(kernel='linear')
            model_rf = RandomForestClassifier()
            model_knn = KNeighborsClassifier(n_neighbors=1)
            model_dt = DecisionTreeClassifier()

            model_svm.fit(tr_features, tr_target)
            model_ada.fit(tr_features, tr_target)
            model_lr.fit(tr_features, tr_target)
            model_lsvm.fit(tr_features, tr_target)
            model_rf.fit(tr_features, tr_target)
            model_knn.fit(tr_features, tr_target)
            model_dt.fit(tr_features, tr_target)

            svm_tst_prediction = model_svm.predict(tst_features)
            ada_tst_prediction = model_ada.predict(tst_features)
            lr_tst_prediction = model_lr.predict(tst_features)
            lsvm_tst_prediction = model_lsvm.predict(tst_features)
            rf_tst_prediction = model_rf.predict(tst_features)
            knn_tst_prediction = model_knn.predict(tst_features)
            dt_tst_prediction = model_dt.predict(tst_features)

            tst_target = np.array(tst_target)

            svm_tst_prediction = np.array(svm_tst_prediction)
            ada_tst_prediction = np.array(ada_tst_prediction)
            lr_tst_prediction = np.array(lr_tst_prediction)
            lsvm_tst_prediction = np.array(lsvm_tst_prediction)
            rf_tst_prediction = np.array(rf_tst_prediction)
            knn_tst_prediction = np.array(knn_tst_prediction)
            dt_tst_prediction = np.array(dt_tst_prediction)

            for i in range(0, len(test_index)):
                prediction_list = [svm_tst_prediction[i] == tst_target[i], ada_tst_prediction[i] == tst_target[i],
                                   lr_tst_prediction[i] == tst_target[i], \
                                   lsvm_tst_prediction[i] == tst_target[i], rf_tst_prediction[i] == tst_target[i],
                                   knn_tst_prediction[i] == tst_target[i], dt_tst_prediction[i] == tst_target[i], \
                                   ]

                prediction_value = [test_index[i], tst_target[i], svm_tst_prediction[i], ada_tst_prediction[i],
                                    lr_tst_prediction[i], \
                                    lsvm_tst_prediction[i], rf_tst_prediction[i], knn_tst_prediction[i],
                                    dt_tst_prediction[i]]

                prediction_value_list.append(prediction_value)

        ensemble_features = pd.DataFrame(prediction_value_list,
                                         columns=['data_id', 'ground_truth', 'rsvm', 'ada', 'lr', 'lsvm', 'rf', 'knn',
                                                  'dt'])

        ensemble_features.set_index('data_id')

        if output_file != "":
            ensemble_features.to_csv(os.path.join(os.getcwd(), output_file), index=False)

    return ensemble_features



def Get_final_prediction_gridsearch(train_features, train_label, test_features, output_file=""):

    print "Get final prediction........."

    estimators = [('clf', RandomForestClassifier())]
    pipe = Pipeline(estimators)

    param_grid = dict(
        clf=[
            SVC(kernel='rbf', C=1),
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=300),
            LogisticRegression(),
            LinearSVC(C=1, dual=False, loss='squared_hinge', penalty='l1'),
            RandomForestClassifier()
        ]
    )

    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    grid_search = grid_search.fit(train_features, train_label)

    best_clf = grid_search.best_estimator_

    scores = cross_val_score(best_clf, train_features, train_label, cv=5)

    print "The best estimator for the ensemble stage:"
    print best_clf

    print "CV result for each folder: "
    print scores

    print "Average accu: "
    print np.mean(scores)

    final_array = best_clf.predict(test_features)

    print "Prediction for test set: "
    print final_array

    target_names = ['class 0', 'class 1']
    predict_train_label = cross_val_predict(best_clf, train_features, train_label, cv=5)

    print(classification_report(train_label, predict_train_label, target_names=target_names))


    if output_file != "":
        np.savetxt(os.path.join(os.getcwd(), output_file), final_array, fmt='%1.1f')


def output_prediction(pre_trained = True):

    features = pd.read_csv(os.path.join(current_path, 'traindata.csv'), header=None)
    targets = pd.read_csv(os.path.join(current_path, 'trainlabel.csv'), header=None)
    test_features = pd.read_csv(os.path.join(current_path, 'testdata.csv'), header=None)

    ensemble_model = ['rsvm', 'ada', 'lr', 'lsvm', 'rf', 'knn', 'dt']


    features, targets = Load_features_toarray(features), Load_label_toarray(targets)
    test_features = Load_features_toarray(test_features)

    if pre_trained:


        ensemble_features = Get_ensemble_features(features, targets, "train_feature_ensemble.csv", "")
        # ensemble_features = Get_ensemble_features(features, targets, "", "train_feature_ensemble.csv")
        ensemble_features = ensemble_features.set_index('data_id').sort_index()

        print ensemble_features

        # test_ensemble_feautures = Get_test_features(test_features, features, targets, "", 'test_feature_ensemble.csv')
        test_ensemble_feautures = Get_test_features(test_features, features, targets, "test_feature_ensemble.csv", '')

        print test_ensemble_feautures

        ensemble_features = ensemble_features[ensemble_model]
        test_ensemble_feautures = test_ensemble_feautures[ensemble_model]

        Get_final_prediction_gridsearch(ensemble_features, targets, test_ensemble_feautures, 'project1_20384933.csv')
    else:

        pre_processing = Select_best_features(features, targets)

        features = pre_processing.fit_transform(features, targets)

        featuresDF = pd.DataFrame(features)
        featuresDF.to_csv(os.path.join(os.getcwd(),'Saved_select_train_features.csv'), index=False)

        test_features = pre_processing.transform(test_features)

        test_featuresDF = pd.DataFrame(test_features)
        test_featuresDF.to_csv(os.path.join(os.getcwd(),'Saved_select_test_features.csv'), index=False)

        features, test_features = Normalize_train_test_features(features, test_features)


        ensemble_features = Get_ensemble_features(features, targets)
        # ensemble_features = Get_ensemble_features(features, targets, "", "train_feature_ensemble.csv")
        ensemble_features = ensemble_features.set_index('data_id').sort_index()

        print ensemble_features

        # test_ensemble_feautures = Get_test_features(test_features, features, targets, "", 'test_feature_ensemble.csv')
        test_ensemble_feautures = Get_test_features(test_features, features, targets)

        print test_ensemble_feautures

        ensemble_features = ensemble_features[ensemble_model]
        test_ensemble_feautures = test_ensemble_feautures[ensemble_model]

        Get_final_prediction_gridsearch(ensemble_features, targets, test_ensemble_feautures, 'output_test_process.out')




if __name__ == '__main__':
    current_path = os.getcwd()

    output_prediction(pre_trained=True)
    #output_prediction(pre_trained=False)

    # features = pd.read_csv(os.path.join(current_path, 'traindata.csv'), header=None)
    # targets = pd.read_csv(os.path.join(current_path, 'trainlabel.csv'), header=None)
    #
    # test_features = pd.read_csv(os.path.join(current_path, 'testdata.csv'), header=None)
    #
    # ensemble_model = ['rsvm', 'ada', 'lr', 'lsvm', 'rf', 'knn', 'dt']
    #
    # targets = Load_label_toarray(targets)
    #
    # # features, targets = Load_features_toarray(features), Load_label_toarray(targets)
    # #
    # # test_features = Load_features_toarray(test_features)
    # #
    # # pre_processing = Select_best_features(features, targets)
    # #
    # # features = pre_processing.fit_transform(features, targets)
    # # test_features = pre_processing.transform(test_features)
    #
    #
    #
    # # features, test_features = Normalize_train_test_features(features, test_features)
    #
    #
    # ensemble_features = Get_ensemble_features(features, targets, "train_feature_ensemble.csv", "")
    # # ensemble_features = Get_ensemble_features(features, targets, "", "train_feature_ensemble.csv")
    # ensemble_features = ensemble_features.set_index('data_id').sort_index()
    #
    # print ensemble_features
    #
    # # test_ensemble_feautures = Get_test_features(test_features, features, targets, "", 'test_feature_ensemble.csv')
    # test_ensemble_feautures = Get_test_features(test_features, features, targets, "test_feature_ensemble.csv",
    #                                             '')
    #
    # print test_ensemble_feautures
    #
    # ensemble_features = ensemble_features[ensemble_model]
    # test_ensemble_feautures = test_ensemble_feautures[ensemble_model]
    #
    # Get_final_prediction_gridsearch(ensemble_features, targets, test_ensemble_feautures, 'output_test4.out')



