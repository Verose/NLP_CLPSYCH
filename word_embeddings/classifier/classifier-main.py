import logging
import os
import warnings

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
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
    def __init__(self, classifier_name,
                 train_accuracy, train_precision, train_recall, train_fscore,
                 test_accuracy, test_precision, test_recall, test_fscore,
                 ):
        self.classifier_name = classifier_name
        self.train_accuracy = train_accuracy
        self.train_precision = train_precision
        self.train_recall = train_recall
        self.train_fscore = train_fscore
        self.test_accuracy = test_accuracy
        self.test_precision = test_precision
        self.test_recall = test_recall
        self.test_fscore = test_fscore


class ClassifierResults:
    def __init__(self, tested_features, num_features, results_list):
        self.tested_features = tested_features
        self.num_features = num_features
        self.results_list = results_list


def classify(model, params):
    gcv = GridSearchCV(model, params, cv=5, scoring="accuracy", iid=False)
    gcv.fit(X_train, y_train)
    best_model = gcv.best_estimator_

    y_hat_train = best_model.predict(X_train)
    y_hat_test = best_model.predict(X_test)
    classifier_name = best_model.__class__.__name__

    train_acc = accuracy_score(y_train, y_hat_train)
    test_acc = accuracy_score(y_test, y_hat_test)
    train_pre, train_recall, train_fscore, _ = precision_recall_fscore_support(
        y_train,
        y_hat_train,
        average='binary'
    )
    test_pre, test_recall, test_fscore, _ = precision_recall_fscore_support(
        y_test,
        y_hat_test,
        average='binary'
    )

    if DEBUG:
        LOGGER.info("************* {}: {} *************".format(classifier_name, test_name))
        LOGGER.info("With best params:", gcv.best_params_)
        LOGGER.info("Train accuracy {}, Test accuracy {}".format(train_acc, test_acc))
        LOGGER.info("Train precision: {}, recall: {}, f-scores: {}".format(train_pre, train_recall, train_fscore))
        LOGGER.info("Test precision: {}, recall: {}, f-scores: {}".format(test_pre, test_recall, test_fscore))

    return classifier_name, \
        train_acc, train_pre, train_recall, train_fscore, \
        test_acc, test_pre, test_recall, test_fscore


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
        # classify pos tags features
        X = get_features(data, test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
    headers = ['Features (#features)',
               'Train Acc', 'Test Acc',
               'Train Precision', 'Test Precision',
               'Train Recall', 'Test Recall',
               'Train F-Score', 'Test F-Score']
    for i in results:
        df_res = pd.DataFrame([('{} ({}) {}'.format(i.tested_features, i.num_features, item.classifier_name),
                                item.train_accuracy, item.test_accuracy,
                                item.train_precision, item.test_precision,
                                item.train_recall, item.test_recall,
                                item.train_fscore, item.test_fscore)
                               for item in i.results_list], columns=headers)
        dfs_res += [df_res]

    dfs_res = pd.concat(dfs_res, axis=0)
    dfs_res.to_csv(os.path.join(OUTPUT_DIR, "classifier.csv"), index=False)
