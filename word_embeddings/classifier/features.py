import os
import string

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import DATA_DIR

pos_tags_features = ['noun',
                     'verb',
                     'adjective',
                     'adverb']
cos_sim_features = [
    'cos_sim'
]
sentiment_features = [
    'positive_count',
    'negative_count'
]
word_related_features = [
    'unique_lemma',
    'unique_tokens'
]
tf_idf_features = [
    'tf_idf'
]


boosting_params = {"n_estimators": [20, 50, 100, 200, 300, 400], "max_depth": [1, 2, 3, 5, 8, 10],
                   "learning_rate": [0.01, 0.03, 0.05]}
rf_params = {"n_estimators": [20, 50, 100, 200, 300, 400], "max_depth": [1, 2, 3, 5],
             "max_features": [1, None, "sqrt"]}
svm_params = {"loss": ['hinge', 'squared_hinge'], "C": [0.5, 1.0, 10], "max_iter": [10000]}


# boosting_params = {"n_estimators": [20], "max_depth": [1],
#                    "learning_rate": [0.01]}
# rf_params = {"n_estimators": [20], "max_depth": [1],
#              "max_features": [1]}
# svm_params = {"loss": ['hinge'], "C": [0.5], "max_iter": [10000]}


def get_answers(removed_ids):
    answers = pd.read_csv(os.path.join(DATA_DIR, 'all_data.csv'))
    answers = answers[~answers['id'].isin(removed_ids)]
    answers = answers.iloc[:, 2:]

    # remove punctuation
    for answer in answers:
        answers[answer] = answers[answer].str.replace('[{}]'.format(string.punctuation), '')

    return answers


def add_tfidf_feature_to_data(orig_df, columns, v):
    X_td = v.fit_transform(columns)

    tf_idf = pd.DataFrame(X_td.toarray(), columns=v.get_feature_names())
    tf_idf = tf_idf.add_prefix('tf_idf-')
    tf_idf = orig_df.reset_index(drop=True).join(tf_idf.reset_index(drop=True), rsuffix='_r')
    return tf_idf


def add_tfidf_features(data, removed_ids):
    answers = get_answers(removed_ids)
    all_answers = answers['question 1']
    v = TfidfVectorizer(lowercase=False)

    for answer in answers.iloc[:, 1:]:
        all_answers = all_answers + " " + answers[answer].fillna('')

    data = add_tfidf_feature_to_data(data, all_answers, v)
    return data


def get_features(df_input, patterns, ans=None):
    if not ans:
        ans = '[0-9]+'
    features = pd.DataFrame()
    for pattern in patterns:
        if pattern != 'tf_idf':
            regex = 'q_{}_{}.*'.format(ans, pattern)
        else:
            regex = pattern
        cols = df_input.filter(regex=regex, axis=1)
        features = pd.concat([features, cols], axis=1)
    return features
