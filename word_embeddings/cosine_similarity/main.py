import logging
import warnings

from word_embeddings.cosine_similarity.cosine_similarity import CosineSimilarity
from word_embeddings.cosine_similarity.ttest_records import WindowTTest, TTest
from word_embeddings.cosine_similarity.utils import *

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

OUTPUT_DIR = os.path.join('..', 'outputs')
LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'main_outputs.txt'))
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def average_cosine_similarity_several_window_sizes():
    win_tests = []

    for win_size in range(1, conf['size'] + 1):
        win_test = WindowTTest(win_size, [])

        cosine_calcs = CosineSimilarity(model, data, conf['mode'], conf['extras'], win_size, data_dir)
        cosine_calcs.init_calculations()

        if conf['mode'] == 'pos' and win_size == 1:
            control_items = cosine_calcs.calculate_items_for_group('control')
            patients_items = cosine_calcs.calculate_items_for_group('patients')
            LOGGER.info('Showing results using POS tags: {}'.format(conf['extras']['pos']))
            LOGGER.info('Average valid words per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_items, patients_items))

        control_score = cosine_calcs.calculate_avg_score_for_group('control')
        patients_score = cosine_calcs.calculate_avg_score_for_group('patients')
        LOGGER.info('\nScores for window size {}: '.format(win_size))
        LOGGER.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))

        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores()
        LOGGER.info('t-test scores on averages: t-statistic: {0:.4f}, p-value: {1:.4f} '.format(tstatistic, pvalue))

        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores_all()
        for i, (stat, p) in enumerate(zip(tstatistic, pvalue), 1):
            win_test.questions_list.append(TTest(i, stat, p))
        win_tests.append(win_test)

        if conf['plot']:
            control_scores_by_question, patient_scores_by_question = cosine_calcs.get_user_to_question_scores()
            plot_groups_scores_by_question(control_scores_by_question, patient_scores_by_question, win_size,
                                           OUTPUT_DIR)
            control_scores, patient_scores = cosine_calcs.get_scores_for_groups()
            plot_groups_histograms(control_scores, patient_scores, win_size, OUTPUT_DIR)

        if conf['mode'] == 'pos':
            control_repetitions = cosine_calcs.calculate_repetitions_for_group('control')
            patients_repetitions = cosine_calcs.calculate_repetitions_for_group('patients')
            LOGGER.info('Average word repetition occurrences per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_repetitions, patients_repetitions))

    LOGGER.debug('t-test results: {}'.format(str(win_tests)))
    ttest_results_to_csv(win_tests, OUTPUT_DIR, conf['extras']['pos'], LOGGER)


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    data = get_medical_data(data_dir)
    conf = read_conf()
    model = FastText.load_fasttext_format(os.path.join(data_dir, 'FastText-pretrained-hebrew', 'wiki.he.bin'))

    average_cosine_similarity_several_window_sizes()
