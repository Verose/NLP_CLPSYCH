import logging
import sys

from word_embeddings.cosine_similarity.basic_sliding_window import BasicSlidingWindow
from word_embeddings.cosine_similarity.pos_tags_window import POSSlidingWindow

logger = logging.getLogger('CosineSimilarity')
logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class CosineSimilarity:
    def __init__(self, model, data, mode, extras, window_size, data_dir=None):
        self._avg_cosine_sim = None

        if mode == "sliding_window":
            self._avg_cosine_sim = BasicSlidingWindow(model, data, window_size, extras)
        elif mode == "pos":
            self._avg_cosine_sim = POSSlidingWindow(model, data, window_size, extras, data_dir)

    def init_calculations(self):
        return self._avg_cosine_sim.calculate_all_avg_scores()

    def get_user_to_question_scores(self):
        control_scores = self._avg_cosine_sim.get_user_to_question_scores('control')
        patients_scores = self._avg_cosine_sim.get_user_to_question_scores('patients')
        return control_scores, patients_scores

    def get_scores_for_groups(self):
        control_scores = self._avg_cosine_sim.get_scores('control')
        patients_scores = self._avg_cosine_sim.get_scores('patients')
        return control_scores, patients_scores

    def calculate_avg_score_for_group(self, group='control'):
        return self._avg_cosine_sim.calculate_avg_score_for_group(group)

    def calculate_ttest_scores(self):
        return self._avg_cosine_sim.perform_ttest_on_averages()

    def calculate_ttest_scores_all(self):
        return self._avg_cosine_sim.perform_ttest_on_all()

    def calculate_repetitions_for_group(self, group='control'):
        return self._avg_cosine_sim.calculate_repetitions_for_group(group)

    def calculate_items_for_group(self, group='control'):
        return self._avg_cosine_sim.calculate_items_for_group(group)
