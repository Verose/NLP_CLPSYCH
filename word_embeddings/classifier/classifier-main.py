import optparse
import os
import warnings

import pandas as pd

from classifier_tests import classify_base
from features import pos_tags_features, cos_sim_features, sentiment_features, word_related_features, get_features
from utils import DATA_DIR, LOGGER

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--debug', action="store_true", default=False)
    options, remainder = parser.parse_args()

    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    # only keep male users
    df_res = df_res[df_res['gender'] == 1]
    # only keep control and schizophrenia groups
    df_res = df_res[(df_res['diagnosys_group'] == 'control') | (df_res['diagnosys_group'] == 'schizophrenia')]
    y = df_res['label'].replace({'control': 0, 'patient': 1})
    all_features = pos_tags_features + cos_sim_features + sentiment_features + word_related_features
    data = get_features(df_res, all_features)

    tests = [
        (pos_tags_features, 'PosTags'),
        (cos_sim_features, 'CosSim'),
        (sentiment_features, 'Sentiment'),
        (word_related_features, 'WordRelated'),
        (cos_sim_features + sentiment_features, 'CosSim + Sentiment'),
        (pos_tags_features + sentiment_features + word_related_features, 'PosTags + Sentiment + WordRelated'),
        (all_features, 'All Features')
    ]

    LOGGER.info('-----------------------------------------------------------------')
    LOGGER.info('--------------------- Performing Base Tests ---------------------')
    LOGGER.info('-----------------------------------------------------------------')
    classify_base(data, y, tests, options.debug)
