import logging
import sys

from tqdm import tqdm

from word_embeddings.common.utils import read_conf, load_model, get_words
from word_embeddings.cosine_similarity.cos_sim_records import WindowCosSim, CosSim
from word_embeddings.cosine_similarity.pos_tags_window import POSSlidingWindow
from word_embeddings.cosine_similarity.ttest_records import WindowTTest, TTest
from word_embeddings.cosine_similarity.utils import *

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUTS_DIR, 'main_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def cosine_similarity_several_window_sizes(window_sizes):
    win_ttests = []
    win_cossim = []
    groups_diff_gs = []
    control_scores_gs = []
    patients_scores_gs = []
    ttest_scores_gs = []

    for i, win_size in tqdm(enumerate(window_sizes), file=sys.stdout, total=len(window_sizes), leave=False,
                            desc='Window Sizes'):
        pos_tags = conf['pos_tags'][i]
        cosine_calcs = POSSlidingWindow(model, data, win_size, DATA_DIR, pos_tags, conf['questions'],
                                        conf['question_minimum_length'])
        cosine_calcs.calculate_all_scores()

        win_ttest = WindowTTest(pos_tags, win_size, [])
        cossim_test = WindowCosSim(pos_tags, win_size, [])

        control_score = cosine_calcs.calculate_group_scores('control')
        patients_score = cosine_calcs.calculate_group_scores('patients')
        LOGGER.info('Scores for window size {}: '.format(win_size))
        LOGGER.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))

        control_scores_gs.append((win_size, control_score))
        patients_scores_gs.append((win_size, patients_score))
        groups_diff_gs.append((win_size, control_score - patients_score))

        tstatistic, pvalue = cosine_calcs.perform_ttest_on_averages()
        LOGGER.info('t-test scores on averages: t-statistic: {0:.4f}, p-value: {1:.4f} '.format(tstatistic, pvalue))
        ttest_scores_gs.append((win_size, tstatistic, pvalue))

        tstatistic, pvalue = cosine_calcs.perform_ttest_on_all()
        for j, stat, p in zip(conf['questions'], tstatistic, pvalue):
            win_ttest.questions_list.append(TTest(j, stat, p))
        win_ttests.append(win_ttest)

        calculate_cossim_scores_for_test(cosine_calcs, cossim_test)
        win_cossim.append(cossim_test)

        if conf['output']['plot']:
            control_scores_by_question = cosine_calcs.get_user_to_question_scores('control')
            patient_scores_by_question = cosine_calcs.get_user_to_question_scores('patients')
            plot_groups_scores_by_question(control_scores_by_question, patient_scores_by_question, win_size,
                                           pos_tags, OUTPUTS_DIR)

        LOGGER.info('Showing results using POS tags: {}'.format(pos_tags))

        words_without_embeddings = set(cosine_calcs.words_without_embeddings)
        LOGGER.info('Words in corpus without FastText word embeddings: {}'.format(words_without_embeddings))
        LOGGER.info('*************************************')

    LOGGER.debug('t-test results: {}'.format(str(win_ttests)))
    unique_name = '_' + conf['pos_tags'][0] if conf['output']['grid_search'] else '_' + '_'.join(conf['pos_tags'])
    ttest_results_to_csv(win_ttests, LOGGER, unique_name)
    cossim_scores_per_question_to_csv(win_cossim, LOGGER, unique_name)

    ret_val = {
        "groups_diff_gs": groups_diff_gs,
        "control_scores": control_scores_gs,
        "patients_scores": patients_scores_gs,
        "ttest_scores": ttest_scores_gs,
    }

    return ret_val


def calculate_cossim_scores_for_test(cosine_calcs, cossim_test):
    control_scores_by_question = cosine_calcs.get_user_to_question_scores('control')
    patient_scores_by_question = cosine_calcs.get_user_to_question_scores('patients')
    control_v_words_by_question = cosine_calcs.get_user_to_question_valid_words('control')
    patient_v_words_by_question = cosine_calcs.get_user_to_question_valid_words('patients')

    for userid, scores in control_scores_by_question.items():
        valid_words_list = control_v_words_by_question[userid]
        for qnum, score in scores.items():
            if score > -2:
                valid_words = valid_words_list[qnum]
                cossim_test.questions_list.append(CosSim(userid, 'control', qnum, score, valid_words))
    for userid, scores in patient_scores_by_question.items():
        valid_words_list = patient_v_words_by_question[userid]
        for qnum, score in scores.items():
            if score > -2:
                valid_words = valid_words_list[qnum]
                cossim_test.questions_list.append(CosSim(userid, 'patients', qnum, score, valid_words))


if __name__ == '__main__':
    conf = read_conf()
    data = get_medical_data(clean_data=conf["clean_data"])
    data = data[['id', 'label']]
    model = load_model(get_words(), conf['word_embeddings'])

    if conf['output']['grid_search']:
        pos_tags_list = conf['pos_tags']
        grid_search_count = conf['output']['grid_search_count']
        pos_tags_win_sizes = range(1, grid_search_count+1)
        grid_search = []

        for pos_tag in tqdm(pos_tags_list, file=sys.stdout, total=len(pos_tags_list), leave=False, desc='Grid Search'):
            conf['pos_tags'] = [pos_tag] * grid_search_count
            grid_search_window = cosine_similarity_several_window_sizes(pos_tags_win_sizes)
            grid_search.append((pos_tag, grid_search_window["groups_diff_gs"]))
            groups_scores = {
                "control": grid_search_window["control_scores"],
                "patients": grid_search_window["patients_scores"]
            }
            ttest_scores = grid_search_window["ttest_scores"]
            plot_word_categories_to_coherence(groups_scores, ttest_scores, pos_tag)

        if conf['output']['plot']:
            plot_grid_search(grid_search, OUTPUTS_DIR)
    else:
        cosine_similarity_several_window_sizes(conf['window_size'])
