import os
import warnings

import pandas as pd

from classifier_tests import classify_base_best_classifier
from features import pos_tags_features, cos_sim_features, sentiment_features, word_related_features, get_features, \
    add_tfidf_features, tf_idf_features
from utils import DATA_DIR, LOGGER, remove_females, remove_depressed

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


if __name__ == '__main__':
    removed_ids = []
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    df_res = remove_females(df_res, removed_ids)
    df_res = remove_depressed(df_res, removed_ids)
    y = df_res['label'].replace({'control': 0, 'patient': 1})
    text_features = pos_tags_features + word_related_features + tf_idf_features
    all_features = text_features + sentiment_features + cos_sim_features

    data = get_features(df_res, all_features)
    data = add_tfidf_features(data, removed_ids)

    tests = [
        (pos_tags_features, 'PosTags'),
        (cos_sim_features, 'CosSim'),
        (sentiment_features, 'Sentiment'),
        (word_related_features, 'WordRelated'),
        (tf_idf_features, 'TfIdf'),
        (cos_sim_features + pos_tags_features + sentiment_features, 'CosSim + PosTags + Sentiment'),
        (cos_sim_features + tf_idf_features, 'CosSim + TfIdf'),
        (text_features, 'Text Features'),
        (all_features, 'All Features')
    ]

    LOGGER.info('-----------------------------------------------------------------')
    LOGGER.info('------- Performing T-TestsBase of Best Classifier Results -------')
    LOGGER.info('-----------------------------------------------------------------')
    classify_base_best_classifier(data, y, tests)
