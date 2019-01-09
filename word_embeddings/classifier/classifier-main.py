import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics.scorer import recall_scorer, precision_scorer, accuracy_scorer, f1_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'classifier_outputs.txt'))
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


pos_tags_features = ['noun',
                     'verb',
                     'adjective',
                     'adverb']
cos_sim_features = [
    'cos_sim'
]
positivity_features = [
    'positive_count',
    'negative_count'
]


DEBUG = True


class ResultsRecord:
    def __init__(self, classifier_name, accuracy, precision, recall, f1):
        self.classifier_name = classifier_name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1


class ClassifierResults:
    def __init__(self, tested_features, num_features, results_list):
        self.tested_features = tested_features
        self.num_features = num_features
        self.results_list = results_list


def classify(model, params):
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

    if DEBUG:
        LOGGER.info("************* {}: {} *************".format(classifier_name, test_name))
        LOGGER.info("With best params: {}".format(gcv.best_params_))
        LOGGER.info("accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))

    return classifier_name, accuracy, precision, recall, f1


def get_features(df_input, column):
    features = pd.DataFrame()
    for col in column:
        regex = 'q_[0-9]+_{}.*'.format(col)
        cols = df_input.filter(regex=regex, axis=1)
        features = pd.concat([features, cols], axis=1)
    return features


if __name__ == '__main__':
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    all_features = pos_tags_features + cos_sim_features + positivity_features
    data = get_features(df_res, all_features)
    y = df_res['label'].replace({'control': 0, 'patient': 1})

    tests = [(pos_tags_features, 'PosTags'), (cos_sim_features, 'CosSim'), (positivity_features, 'Positivity'),
             (cos_sim_features + positivity_features, 'CosSim + Positivity'), (all_features, 'All Features')]
    results = []

    for test, test_name in tests:
        X = get_features(data, test)

        test_results = ClassifierResults(test_name, len(X.columns), [])
        xgboost_params = classify(XGBClassifier(),
                                  params={"n_estimators": [20, 50, 100, 200, 300, 400],
                                          "max_depth": [1, 2, 3, 5, 8, 10],
                                          "learning_rate": [0.01, 0.03, 0.05]})
        test_results.results_list.append(ResultsRecord(*xgboost_params))

        gb_params = classify(GradientBoostingClassifier(),
                             params={"n_estimators": [20, 50, 100, 200, 300, 400],
                                     "max_depth": [1, 2, 3, 5, 8, 10],
                                     "learning_rate": [0.01, 0.03, 0.05]})
        test_results.results_list.append(ResultsRecord(*gb_params))

        rf_params = classify(RandomForestClassifier(),
                             params={"n_estimators": [20, 50, 100, 200, 300, 400],
                                     "max_depth": [1, 2, 3, 5],
                                     "max_features": [1, 5, None, "sqrt", ]})
        test_results.results_list.append(ResultsRecord(*rf_params))

        svm = classify(LinearSVC(),
                       params={"loss": ['hinge', 'squared_hinge'],
                               "C": [0.5, 1.0, 10],
                               "max_iter": [10000]})
        test_results.results_list.append(ResultsRecord(*svm))

        results += [test_results]

    # print pretty csv
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', -1)
    dfs_res = []
    headers = ['Features (#features)', 'Acc.', 'Prec.', 'Rec.', 'F']

    for i in results:
        df_res = pd.DataFrame([('{} ({}) {}'.format(i.tested_features, i.num_features, item.classifier_name),
                                item.accuracy, item.precision, item.recall, item.f1)
                               for item in i.results_list], columns=headers)
        dfs_res += [df_res]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier.csv"), index=False)
