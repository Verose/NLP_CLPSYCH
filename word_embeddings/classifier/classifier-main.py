import optparse
import os
import warnings

import pandas as pd

from classifier_tests import classify_base, classify_per_question
from features import pos_tags_features, cos_sim_features, sentiment_features, word_related_features, get_features, \
    add_tfidf_features, tf_idf_features
from utils import DATA_DIR, LOGGER

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def remove_females(df, removed):
    # only keep male users
    res = df[df['gender'] == 1]
    removed.extend(list(df[df['gender'] == 2]['id']))
    return res


def remove_depressed(df, removed):
    # only keep control and schizophrenia groups
    res = df_res[(df_res['diagnosys_group'] == 'control') | (df_res['diagnosys_group'] == 'schizophrenia')]
    removed.extend(list(df[df['diagnosys_group'] == 'depression']['id']))
    return res


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--debug', action="store_true", default=False)
    options, remainder = parser.parse_args()

    removed_ids = []
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    df_res = remove_females(df_res, removed_ids)
    df_res = remove_depressed(df_res, removed_ids)
    y = df_res['label'].replace({'control': 0, 'patient': 1})
    all_features = pos_tags_features + cos_sim_features + sentiment_features + word_related_features
    # all_features = pos_tags_features + cos_sim_features + sentiment_features + word_related_features + tf_idf_features
    data = get_features(df_res, all_features)
    # data = add_tfidf_features(data, removed_ids)

    tests = [
        (pos_tags_features, 'PosTags'),
        (cos_sim_features, 'CosSim'),
        (sentiment_features, 'Sentiment'),
        (word_related_features, 'WordRelated'),
        # (tf_idf_features, 'TfIdf'),
        (cos_sim_features + pos_tags_features, 'CosSim + PosTags'),
        # (cos_sim_features + tf_idf_features, 'CosSim + TfIdf'),
        (pos_tags_features + word_related_features, 'PosTags + WordRelated'),
        # (pos_tags_features + sentiment_features + word_related_features + tf_idf_features, 'All Text Features'),
        (all_features, 'All Features')
    ]

    LOGGER.info('-----------------------------------------------------------------')
    LOGGER.info('--------------------- Performing Base Tests ---------------------')
    LOGGER.info('-----------------------------------------------------------------')
    classify_base(data, y, tests, options.debug)

    LOGGER.info('-----------------------------------------------------------------')
    LOGGER.info('----------------- Performing Per-Question Tests -----------------')
    LOGGER.info('-----------------------------------------------------------------')
    classify_per_question(data, y, tests, options.debug)
