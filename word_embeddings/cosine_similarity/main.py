import logging
import warnings

from word_embeddings.cosine_similarity.cos_sim_records import WindowCosSim, CosSim
from word_embeddings.cosine_similarity.cosine_similarity import CosineSimilarity
from word_embeddings.cosine_similarity.ttest_records import WindowTTest, TTest
from word_embeddings.cosine_similarity.utils import *

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.wrappers import FastText

OUTPUT_DIR = os.path.join('..', 'outputs')
DATA_DIR = os.path.join('..', 'data')
LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUT_DIR, 'main_outputs.txt'))
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def average_cosine_similarity_several_window_sizes(window_sizes):
    win_tests = []
    win_cossim = []
    gs_window = []

    for i, win_size in enumerate(window_sizes):
        if conf['mode'] == 'pos':
            pos_tags = conf['pos_tags'][i]
            header = pos_tags
            window_method = conf['window_method']
            cosine_calcs = CosineSimilarity(model, data, conf['mode'], win_size, DATA_DIR,
                                            pos_tags=pos_tags, window_method=window_method)
        else:
            header = 'default'
            cosine_calcs = CosineSimilarity(model, data, conf['mode'], win_size, DATA_DIR)
        cosine_calcs.init_calculations()

        win_test = WindowTTest(header, win_size, [])
        cossim_test = WindowCosSim(header, win_size, [])

        control_score = cosine_calcs.calculate_avg_score_for_group('control')
        patients_score = cosine_calcs.calculate_avg_score_for_group('patients')
        LOGGER.info('Scores for window size {}: '.format(win_size))
        LOGGER.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))
        gs_window.append((win_size, control_score-patients_score))

        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores()
        LOGGER.info('t-test scores on averages: t-statistic: {0:.4f}, p-value: {1:.4f} '.format(tstatistic, pvalue))

        tstatistic, pvalue = cosine_calcs.calculate_ttest_scores_all()
        for j, (stat, p) in enumerate(zip(tstatistic, pvalue), 1):
            win_test.questions_list.append(TTest(j, stat, p))
        win_tests.append(win_test)

        control_scores_by_question, patient_scores_by_question = cosine_calcs.get_user_to_question_scores()
        for userid, scores in control_scores_by_question.items():
            for qnum, score in scores.items():
                cossim_test.questions_list.append(CosSim(userid, qnum, score))
        for userid, scores in patient_scores_by_question.items():
            for qnum, score in scores.items():
                cossim_test.questions_list.append(CosSim(userid, qnum, score))
        win_cossim.append(cossim_test)

        if conf['output']['plot']:
            plot_groups_scores_by_question(control_scores_by_question, patient_scores_by_question, win_size,
                                           header, OUTPUT_DIR)
            control_scores, patient_scores = cosine_calcs.get_scores_for_groups()
            plot_groups_histograms(control_scores, patient_scores, win_size, header, OUTPUT_DIR)

        if conf['mode'] == 'pos':
            control_items = cosine_calcs.calculate_items_for_group('control')
            patients_items = cosine_calcs.calculate_items_for_group('patients')
            LOGGER.info('Showing results using POS tags: {}'.format(pos_tags))
            LOGGER.info('Average valid words per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_items, patients_items))

            control_repetitions = cosine_calcs.calculate_repetitions_for_group('control')
            patients_repetitions = cosine_calcs.calculate_repetitions_for_group('patients')
            LOGGER.info('Average word repetition occurrences per answer: Control: {0:.4f}, Patients: {1:.4f}'
                        .format(control_repetitions, patients_repetitions))

    LOGGER.debug('t-test results: {}'.format(str(win_tests)))
    ttest_results_to_csv(win_tests, OUTPUT_DIR, LOGGER)
    cossim_scores_to_csv(win_cossim, OUTPUT_DIR, LOGGER)
    return gs_window


if __name__ == '__main__':
    data = get_medical_data(DATA_DIR)
    conf = read_conf(DATA_DIR)
    model = FastText.load_fasttext_format(os.path.join(DATA_DIR, 'ft_pretrained', 'wiki.he.bin'))

    if conf['output']['grid_search'] and conf['mode'] == 'pos':
        pos_tags_list = ['noun', 'verb', 'adjective', 'adverb', 'noun verb adverb adjective', 'verb adverb']
        pos_tags_win_sizes = [conf['output']['grid_search_count']] * len(pos_tags_list)
        grid_search = []

        for pos_tags in pos_tags_list:
            conf['pos_tags'] = [pos_tags]
            grid_search_window = average_cosine_similarity_several_window_sizes(pos_tags_win_sizes)
            grid_search.append((pos_tags, grid_search_window))

        plot_grid_search(grid_search, OUTPUT_DIR)
    else:
        average_cosine_similarity_several_window_sizes(conf['window_size'])
