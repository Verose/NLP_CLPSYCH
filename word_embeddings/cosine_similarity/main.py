import logging
import warnings

from word_embeddings.cosine_similarity.cosine_similarity import CosineSimilarity
from word_embeddings.cosine_similarity.utils import *

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

output_dir = os.path.join('..', 'outputs')
logger = logging.getLogger('Main')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(output_dir, 'main_outputs.txt'))
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def average_cosine_similarity_several_window_sizes():
    for win_size in range(1, conf['size'] + 1):
        cosine_calcs = CosineSimilarity(model, data, conf['mode'], conf['extras'], win_size, data_dir)
        cosine_calcs.init_calculations()

        if conf['mode'] == 'pos' and win_size == 1:
            control_items = cosine_calcs.calculate_items_for_group('control')
            patients_items = cosine_calcs.calculate_items_for_group('patients')
            logger.info('Showing results using POS tags: {}'.format(conf['extras']['pos']))
            logger.info('Average valid words per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_items, patients_items))

        control_score = cosine_calcs.calculate_avg_score_for_group('control')
        patients_score = cosine_calcs.calculate_avg_score_for_group('patients')
        logger.info('\nScores for window size {}: '.format(win_size))
        logger.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))

        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores()
        logger.info('t-test scores on averages: t-statistic: {0:.4f}, p-value: {1:.4f} '.format(tstatistic, pvalue))
        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores_all()
        ttest_all = ''
        for i, (stat, p) in enumerate(zip(tstatistic, pvalue), 1):
            ttest_all += 'Q{0}: t-statistic: {1:.4f}, p-value: {2:.4f} '.format(i, stat, p)
        logger.info(ttest_all)

        if conf['plot']:
            control_scores_by_question, patient_scores_by_question = cosine_calcs.get_user_to_question_scores()
            plot_groups_scores_by_question(control_scores_by_question, patient_scores_by_question, win_size,
                                           output_dir)
            control_scores, patient_scores = cosine_calcs.get_scores_for_groups()
            plot_groups_histograms(control_scores, patient_scores, win_size, output_dir)

        if conf['mode'] == 'pos':
            control_repetitions = cosine_calcs.calculate_repetitions_for_group('control')
            patients_repetitions = cosine_calcs.calculate_repetitions_for_group('patients')
            logger.info('Average word repetition occurrences per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_repetitions, patients_repetitions))


if __name__ == '__main__':
    data_dir = os.path.join('..', 'data')
    data = get_medical_data(data_dir)
    conf = read_conf()
    model = FastText.load_fasttext_format(os.path.join(data_dir, 'FastText-pretrained-hebrew', 'wiki.he.bin'))

    average_cosine_similarity_several_window_sizes()
