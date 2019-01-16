import operator
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics.scorer import accuracy_scorer, precision_scorer, recall_scorer, f1_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from tqdm import tqdm
from xgboost import XGBClassifier

from features import get_features, boosting_params, rf_params, svm_params
from results_records import TestResults, ResultsRecord, AnswersResults
from utils import LOGGER, OUTPUT_DIR

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


scoring = {
    'accuracy': accuracy_scorer,
    'precision': precision_scorer,
    'recall': recall_scorer,
    'f1': f1_scorer,
}


def classify(X, y, model, params, test_name, debug):
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
            important_features.sort(key=operator.itemgetter(1), reverse=True)  # sort by importance
            [LOGGER.info('Feature {}: {}'.format(*feat)) for feat in important_features]

    return classifier_name, accuracy, precision, recall, f1


def get_classifier_result_records(X, y, test_name, debug):
    result_records = []

    # XGBoost classification
    xgboost_results = classify(X, y, XGBClassifier(),
                               params=boosting_params,
                               test_name=test_name,
                               debug=debug)
    result_records.append(ResultsRecord(*xgboost_results))

    # GradientBoosting classification
    gb_results = classify(X, y, GradientBoostingClassifier(),
                          params=boosting_params,
                          test_name=test_name,
                          debug=debug)
    result_records.append(ResultsRecord(*gb_results))

    # RandomForest classification
    rf_results = classify(X, y, RandomForestClassifier(),
                          params=rf_params,
                          test_name=test_name,
                          debug=debug)
    result_records.append(ResultsRecord(*rf_results))

    # SVM classification
    svm_results = classify(X, y, LinearSVC(),
                           params=svm_params,
                           test_name=test_name,
                           debug=debug)
    result_records.append(ResultsRecord(*svm_results))

    return result_records


def classify_base(data, y, tests, debug):
    results = []

    for test, test_name in tqdm(tests, file=sys.stdout, total=len(tests)):
        X = get_features(data, test)
        test_results = TestResults(test_name, len(X.columns), [])
        result_records = get_classifier_result_records(X, y, test_name, debug)

        for record in result_records:
            test_results.results_list.append(record)

        results += [test_results]

    dfs_res = []
    headers = ['Features (#features)', 'Acc.', 'Prec.', 'Recall', 'F']

    for i in results:
        df_res = pd.DataFrame([('{} ({}) {}'.format(i.tested_features, i.num_features, item.classifier_name),
                                item.accuracy, item.precision, item.recall, item.f1)
                               for item in i.results_list],
                              columns=headers)
        dfs_res += [df_res]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier_base.csv"), index=False)


def classify_per_question(data, y, tests, debug):
    results = []

    for test, test_name in tqdm(tests, file=sys.stdout, total=len(tests)):
        test_results = TestResults(test_name, len(test), [])

        for ans in range(1, 19):
            X = get_features(data, test, str(ans))
            answer_results = AnswersResults(str(ans), [])
            result_records = get_classifier_result_records(X, y, test_name + ' q{}'.format(str(ans)), debug)

            for record in result_records:
                answer_results.results_list.append(record)

            test_results.results_list.append(answer_results)
        results += [test_results]

    dfs_res = []
    headers = ['Acc. q{}', 'Prec. q{}', 'Recall q{}', 'F q{}']
    features_list = []

    for result in results:
        df_tests = []

        # iterate classifier names, all answers use the same classifiers. choose 0
        for cls in result.results_list[0].results_list:
            feat_head = '{} ({}) {}'.format(result.tested_features, result.num_features, cls.classifier_name)
            features_list.append(feat_head)

        for item in result.results_list:
            ans_headers = [head.format(item.answer_number) for head in headers]
            df_test = pd.DataFrame([(i.accuracy, i.precision, i.recall, i.f1)
                                   for i in item.results_list],
                                   columns=ans_headers)
            df_tests += [df_test]

        df_tests = pd.concat(df_tests, axis=1)
        dfs_res += [df_tests]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.insert(0, 'Features (#features)', features_list)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier_per_answer.csv"), index=False)


def classify_question_types(data, y, tests, debug):
    results = []

    for test, test_name in tqdm(tests, file=sys.stdout, total=len(tests)):
        test_results = TestResults(test_name, len(test), [])
        answer_ranges = [('([1-9]|1[0-4])', '1-14'), ('(1[5-8])', '15-18')]

        for regex, ans_range in answer_ranges:
            X = get_features(data, test, regex)
            answer_results = AnswersResults(ans_range, [])
            result_records = get_classifier_result_records(X, y, test_name + ' q{}'.format(ans_range), debug)

            for record in result_records:
                answer_results.results_list.append(record)

            test_results.results_list.append(answer_results)
        results += [test_results]

    dfs_res = []
    headers = ['Acc. q{}', 'Prec. q{}', 'Recall q{}', 'F q{}']
    features_list = []

    for result in results:
        df_tests = []

        # iterate classifier names, all answer_ranges use the same classifiers. choose 0
        for cls in result.results_list[0].results_list:
            feat_head = '{} ({}) {}'.format(result.tested_features, result.num_features, cls.classifier_name)
            features_list.append(feat_head)

        for item in result.results_list:
            ans_headers = [head.format(item.answer_number) for head in headers]
            df_test = pd.DataFrame([(i.accuracy, i.precision, i.recall, i.f1)
                                   for i in item.results_list],
                                   columns=ans_headers)
            df_tests += [df_test]

        df_tests = pd.concat(df_tests, axis=1)
        dfs_res += [df_tests]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.insert(0, 'Features (#features)', features_list)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier_answer_type.csv"), index=False)
