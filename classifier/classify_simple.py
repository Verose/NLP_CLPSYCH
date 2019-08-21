import operator
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.scorer import accuracy_scorer, precision_scorer, recall_scorer, f1_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from utils import LOGGER
from common.utils import DATA_DIR, OUTPUTS_DIR

scoring = {
    'accuracy': accuracy_scorer,
    'precision': precision_scorer,
    'recall': recall_scorer,
    'f1': f1_scorer,
}

tree_params = {
    "n_estimators": [1, 3, 5, 10, 15, 20, 50, 100, 200],
    "max_depth": [1, 2, 3, 5, 8, 10, 20],
    "max_features": [1, None, "sqrt"]
}
svm_params = {
    "loss": ['hinge', 'squared_hinge'],
    "C": [0.5, 1.0, 10, 100, 1000],
    "max_iter": [1000000]
}


def classify_cv_results(X, y, model, params, debug, cv=5):
    gcv = GridSearchCV(model, params, cv=cv, scoring=scoring, refit='accuracy', iid=False)
    gcv.fit(X, y)
    best_model = gcv.best_estimator_
    classifier_name = best_model.__class__.__name__
    accuracy = np.mean(gcv.cv_results_['mean_test_accuracy'])
    precision = np.mean(gcv.cv_results_['mean_test_precision'])
    recall = np.mean(gcv.cv_results_['mean_test_recall'])
    f1 = np.mean(gcv.cv_results_['mean_test_f1'])

    if debug:
        LOGGER.info("************* {} *************".format(classifier_name))
        LOGGER.info("With best params: {}".format(gcv.best_params_))
        LOGGER.info("accuracy: {}, precision: {}, recall: {}, f1: {}".format(accuracy, precision, recall, f1))

        if hasattr(best_model, 'feature_importances_'):
            important_features = [(X.columns[i], importance)
                                  for i, importance in enumerate(best_model.feature_importances_)
                                  if importance > 0]
            LOGGER.info('Important features: ')
            important_features.sort(key=operator.itemgetter(1), reverse=True)  # sort by importance
            [LOGGER.info('Feature {}: {}'.format(*feat)) for feat in important_features]

    return classifier_name, accuracy, precision, recall, f1


if __name__ == '__main__':
    scores_all = pd.read_csv(os.path.join(DATA_DIR, 'scores_all.csv'))
    scores_content = pd.read_csv(os.path.join(DATA_DIR, 'scores_content.csv'))
    scores_dependency = pd.read_csv(os.path.join(DATA_DIR, 'scores_dependency.csv'))

    scores = scores_all.merge(scores_content, on=['user', 'Group', 'Gender'], suffixes=('_all', '_content'))
    scores = scores.merge(scores_dependency, on=['user'])

    y = scores['Group'].replace({'control': 0, 'patients': 1})
    X = scores.drop(columns=['Group', 'user'])

    # RandomForest classification
    rf_results = classify_cv_results(X, y, RandomForestClassifier(),
                                     params=tree_params,
                                     debug=True,
                                     cv=10)

    headers = ['Acc.', 'Prec.', 'Recall', 'F1']
    df_res = pd.DataFrame([[*rf_results[1:]]], columns=headers)
    df_res.to_csv(os.path.join(OUTPUTS_DIR, "classifier_basic_rf.csv"), index=False)

    # XGBClassifier classification
    xgb_results = classify_cv_results(X, y, XGBClassifier(),
                                      params=tree_params,
                                      debug=True,
                                      cv=10)

    headers = ['Acc.', 'Prec.', 'Recall', 'F1']
    df_res = pd.DataFrame([[*xgb_results[1:]]], columns=headers)
    df_res.to_csv(os.path.join(OUTPUTS_DIR, "classifier_basic_xgb.csv"), index=False)

    # Linear SVM classification
    svm_results = classify_cv_results(X, y, LinearSVC(),
                                      params=svm_params,
                                      debug=True,
                                      cv=10)

    headers = ['Acc.', 'Prec.', 'Recall', 'F1']
    df_res = pd.DataFrame([[*svm_results[1:]]], columns=headers)
    df_res.to_csv(os.path.join(OUTPUTS_DIR, "classifier_basic_svm.csv"), index=False)
