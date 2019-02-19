import logging
import os

import pandas as pd

from word_embeddings.common.utils import remove_females, remove_depressed, DATA_DIR, OUTPUTS_DIR
from word_embeddings.dependency_parser.dependency_cos_sim import DependencyCosSimScorer


LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUTS_DIR, 'dependency_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def main():
    removed_ids = []
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    df_res = remove_females(df_res, removed_ids)
    df_res = remove_depressed(df_res, removed_ids)
    control = (df_res[df_res['label'] == 'control']['id'].values, 'control')
    patients = (df_res[df_res['label'] == 'patient']['id'].values, 'patients')

    dep_scorer = DependencyCosSimScorer()

    users_scores = {}
    for group, group_name in [control, patients]:
        for user in group:
            user_scores = dep_scorer.cos_sim_score_for_user(user, group_name)
            users_scores[user] = user_scores

    avgs_nouns, avgs_verbs = dep_scorer.all_users_scores(control[0], users_scores)
    control_scores = dep_scorer.score_group(avgs_nouns, avgs_verbs)
    control_users_scores = dep_scorer.save_per_user_scores(avgs_nouns, avgs_verbs, *control)
    LOGGER.info("Control group scores: nouns: {}, verbs: {}".format(*control_scores))

    avgs_nouns, avgs_verbs = dep_scorer.all_users_scores(patients[0], users_scores)
    patients_scores = dep_scorer.score_group(avgs_nouns, avgs_verbs)
    patients_users_scores = dep_scorer.save_per_user_scores(avgs_nouns, avgs_verbs, *patients)
    LOGGER.info("Patients group scores: nouns: {}, verbs: {}".format(*patients_scores))

    ttest_results = dep_scorer.ttest_group_scores(control_users_scores, patients_users_scores)
    pvalues = ttest_results.pvalue
    tstatistics = ttest_results.statistic
    LOGGER.info('T-Test results:')
    LOGGER.info('nouns: p-value: {}, t-statistic: {}'.format(pvalues[0], tstatistics[0]))
    LOGGER.info('verbs: p-value: {}, t-statistic: {}'.format(pvalues[1], tstatistics[1]))
    LOGGER.info('No idf scores for words {}\n'.format(dep_scorer.missing_idf))
    LOGGER.info('Words without embeddings {}\n'.format(set(dep_scorer.words_without_embeddings)))


if __name__ == '__main__':
    main()
