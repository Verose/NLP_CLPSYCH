import datetime
import json
import logging
import sys

from tqdm import tqdm

from word_embeddings.common.utils import read_conf
from word_embeddings.cosine_similarity.cos_sim_records import WindowCosSim
from word_embeddings.cosine_similarity.pos_tags_window import POSSlidingWindow
from word_embeddings.cosine_similarity.utils import *

LOGGER = logging.getLogger('Main')
LOGGER.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename=os.path.join(OUTPUTS_DIR, 'main_outputs.txt'), encoding='utf-8')
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)


class PosTagsGenerator:
    def __init__(self, data_dir, pos_tags_folder, num_posts_limit=500):
        self._data_dir = data_dir
        self._pos_tags_folder = pos_tags_folder
        self._num_posts_limit = num_posts_limit

    def __call__(self, user_id):
        """
        Generator for pos tags.
        :param user_id: user id
        For the hebrew dataset, the pos tags data is read from json files.
        For the english dataset, the pos tags are calculated from the posts.
        :return: (answer number, dictionary with 'tokens' list, and 'posTags' list)
        """
        with open(
                os.path.join(self._data_dir, self._pos_tags_folder, '{}.json'.format(user_id)), encoding='utf-8') as f:
            ans_pos_tags = json.load(f)
            tokens_list = ans_pos_tags['tokens']
            pos_tags_list = ans_pos_tags['posTags']

            for answer_num, (tokens, pos_tags) in enumerate(zip(tokens_list, pos_tags_list), 1):
                if answer_num > self._num_posts_limit:
                    break

                yield answer_num, {'tokens': tokens, 'posTags': pos_tags}


def cosine_similarity_several_window_sizes(window_sizes):
    win_cossim = []
    control_scores_gs = []
    patients_scores_gs = []
    ttest_scores_gs = []

    for i, win_size in tqdm(enumerate(window_sizes), file=sys.stdout, total=len(window_sizes), leave=False,
                            desc='Window Sizes'):
        pos_tags = conf['pos_tags'][i]
        answers_pos_tags_generator = PosTagsGenerator(DATA_DIR, conf['run_params']['pos_tags_folder'],
                                                      conf['run_params']['num_posts_limit'])
        cosine_calcs = POSSlidingWindow(data, win_size, DATA_DIR, pos_tags, conf['run_params'],
                                        answers_pos_tags_generator, pos_tags_type='stanford')
        cosine_calcs.calculate_all_scores()

        cossim_test = WindowCosSim(pos_tags, win_size, [])

        control_score = cosine_calcs.calculate_group_scores('control')
        patients_score = cosine_calcs.calculate_group_scores('patients')
        LOGGER.info('Scores for window size {}: '.format(win_size))
        LOGGER.info('Average Cos-Sim per answer: Control: {0:.4f}, Patients: {1:.4f}'
                    .format(control_score, patients_score))

        control_scores_gs.append((win_size, control_score))
        patients_scores_gs.append((win_size, patients_score))

        tstatistic, pvalue = cosine_calcs.perform_ttest_on_averages()
        LOGGER.info('t-test scores on averages: t-statistic: {0:.4f}, p-value: {1:.4f} '.format(tstatistic, pvalue))
        ttest_scores_gs.append((win_size, tstatistic, pvalue))

        win_cossim.append(cossim_test)

        LOGGER.info('Showing results using POS tags: {}'.format(pos_tags))
        LOGGER.info('*************************************')

    ret_val = {
        "control_scores": control_scores_gs,
        "patients_scores": patients_scores_gs,
        "ttest_scores": ttest_scores_gs,
    }

    return ret_val


if __name__ == '__main__':
    LOGGER.info('Starting: ' + str(datetime.datetime.now()))
    conf = read_conf('medical_rsdd.json')
    data = get_rsdd_data(conf['data_file'])
    pos_tags_list = conf['pos_tags']
    grid_search_count = conf['output']['grid_search_count']
    pos_tags_win_sizes = range(1, grid_search_count+1)

    for pos_tag in tqdm(pos_tags_list, file=sys.stdout, total=len(pos_tags_list), leave=False, desc='Grid Search'):
        conf['pos_tags'] = [pos_tag] * grid_search_count
        grid_search_window = cosine_similarity_several_window_sizes(pos_tags_win_sizes)
        groups_scores = {
            "control": grid_search_window["control_scores"],
            "patients": grid_search_window["patients_scores"]
        }
        ttest_scores = grid_search_window["ttest_scores"]
        plot_word_categories_to_coherence(groups_scores, ttest_scores, pos_tag)
    LOGGER.info('Finished: ' + str(datetime.datetime.now()))
