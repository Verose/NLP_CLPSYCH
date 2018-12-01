import abc
import logging
import os
import sys
from datetime import datetime
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from scipy.stats import stats


class SlidingWindow(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, data, window_size, extras=None):
        self._model = model
        self._data = data
        self._window_size = window_size
        self._extras = extras
        self._logger = self._setup_logger()

        self._control_scores = []
        self._patient_scores = []

    def _setup_logger(self):
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        logger_name = os.path.join(
            '..', 'logs', 'CosSim_{}_{:%m%d%Y-%H%M%S}.txt'.format(self.__class__.__name__, datetime.now()))
        file_handler = RotatingFileHandler(logger_name, backupCount=5, encoding='utf-8')
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", '%H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def get_scores(self, group='control'):
        if group == 'control':
            return [score[0] for score in self._control_scores]
        else:
            return [score[0] for score in self._patient_scores]

    def perform_ttest_on_averages(self):
        """
        Performs t-test on the average cos-sim score of each user
        :return:
        """
        control_scores = self.get_scores('control')
        patient_scores = self.get_scores('patients')
        ttest = stats.ttest_ind(control_scores, patient_scores)
        return ttest

    def perform_ttest_on_all(self):
        """
        Performs t-test on all of the cos-sim scores of each user
        :return:
        """
        control_scores_by_question, patient_scores_by_question = self.get_user_to_question_scores()
        control = [cont for cont in control_scores_by_question.values()]
        patients = [pat for pat in patient_scores_by_question.values()]

        all_control = []
        for cont_dict in control:
            small_control = []
            for cont in cont_dict.values():
                small_control += [cont]
            all_control += [small_control]

        all_pat = []
        for pat_dict in patients:
            small_pat = []
            for pat in pat_dict.values():
                small_pat += [pat]
            all_pat += [small_pat]

        ttest = stats.ttest_ind(all_control, all_pat)
        return ttest

    @abc.abstractmethod
    def calculate_all_avg_scores(self):
        """
        Calculate the cosine similarity of the entire corpus
        :return: dictionary from user id to (average user score, label)
        """
        pass

    @abc.abstractmethod
    def calculate_avg_score_for_group(self, group='control'):
        """
        Calculates the average score for a given group
        :param group: group name, either 'control' or 'patients'
        :return: average score (float)
        """
        pass
