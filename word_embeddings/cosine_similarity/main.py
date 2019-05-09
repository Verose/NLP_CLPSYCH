import logging
import sys

from tqdm import tqdm

from word_embeddings.common.utils import read_conf, load_model, get_words
from word_embeddings.cosine_similarity.pos_tags_window import POSSlidingWindow
from word_embeddings.cosine_similarity.utils import *

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUTS_DIR, 'main_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


def cosine_similarity_several_window_sizes(window_sizes):
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

        LOGGER.info('Showing results using POS tags: {}'.format(pos_tags))
        LOGGER.info('*************************************')

    ret_val = {
        "groups_diff_gs": groups_diff_gs,
        "control_scores": control_scores_gs,
        "patients_scores": patients_scores_gs,
        "ttest_scores": ttest_scores_gs,
    }

    return ret_val


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
