import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics.scorer import accuracy_scorer, precision_scorer, recall_scorer, f1_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from features import get_features
from results_records import ClassifierResults, ResultsRecord
from utils import LOGGER, OUTPUT_DIR

boosting_params = {"n_estimators": [20, 50, 100, 200, 300, 400], "max_depth": [1, 2, 3, 5, 8, 10],
                   "learning_rate": [0.01, 0.03, 0.05]}
rf_params = {"n_estimators": [20, 50, 100, 200, 300, 400], "max_depth": [1, 2, 3, 5],
             "max_features": [1, 5, None, "sqrt", ]}
svm_params = {"loss": ['hinge', 'squared_hinge'], "C": [0.5, 1.0, 10], "max_iter": [10000]}


def classify(X, y, model, params, test_name, debug):
    scoring = {
        'accuracy': accuracy_scorer,
        'precision': precision_scorer,
        'recall': recall_scorer,
        'f1': f1_scorer,
    }
    gcv = GridSearchCV(model, params, cv=5, scoring=scoring, refit='accuracy', iid=False)
    gcv.fit(X, y)
    best_model = gcv.best_estimator_
    classifier_name = best_model.__class__.__name__
    accuracy = np.mean(gcv.cv_results_['mean_test_accuracy'])
    precision = np.mean(gcv.cv_results_['mean_test_precision'])
    recall = np.mean(gcv.cv_results_['mean_test_recall'])
    f1 = np.mean(gcv.cv_results_['mean_test_f1'])

    if debug:
        LOGGER.info("************* {}: {} *************".format(classifier_name, test_name))
        LOGGER.info("With best params: {}".format(gcv.best_params_))
        LOGGER.info("accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))

        if hasattr(best_model, 'feature_importances_'):
            important_features = [(X.columns[i], importance)
                                  for i, importance in enumerate(best_model.feature_importances_)
                                  if importance > 0]
            LOGGER.info('Important features: ')
            [LOGGER.info('Feature {}: {}'.format(*feat)) for feat in important_features]

    return classifier_name, accuracy, precision, recall, f1


def classify_base(data, y, tests, debug):
    results = []

    for test, test_name in tests:
        X = get_features(data, test)
        test_results = ClassifierResults(test_name, len(X.columns), [])

        # XGBoost classification
        xgboost_results = classify(X, y, XGBClassifier(),
                                   params=boosting_params,
                                   test_name=test_name,
                                   debug=debug)
        test_results.results_list.append(ResultsRecord(*xgboost_results))

        # GradientBoosting classification
        gb_results = classify(X, y, GradientBoostingClassifier(),
                              params=boosting_params,
                              test_name=test_name,
                              debug=debug)
        test_results.results_list.append(ResultsRecord(*gb_results))

        # RandomForest classification
        rf_results = classify(X, y, RandomForestClassifier(),
                              params=rf_params,
                              test_name=test_name,
                              debug=debug)
        test_results.results_list.append(ResultsRecord(*rf_results))

        # SVM classification
        svm_results = classify(X, y, LinearSVC(),
                               params=svm_params,
                               test_name=test_name,
                               debug=debug)
        test_results.results_list.append(ResultsRecord(*svm_results))

        results += [test_results]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    dfs_res = []
    headers = ['Features (#features)', 'Acc.', 'Prec.', 'Recall', 'F']

    for i in results:
        df_res = pd.DataFrame([('{} ({}) {}'.format(i.tested_features, i.num_features, item.classifier_name),
                                item.accuracy, item.precision, item.recall, item.f1)
                               for item in i.results_list], columns=headers)
        dfs_res += [df_res]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier_base.csv"), index=False)
