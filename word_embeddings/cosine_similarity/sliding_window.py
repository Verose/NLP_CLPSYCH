import abc
import logging
import os
import sys
from datetime import datetime
from logging import StreamHandler
from logging.handlers import RotatingFileHandler


class SlidingWindow(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, data, window_size, extras=None):
        self._model = model
        self._data = data
        self._window_size = window_size
        self._extras = extras
        self._logger = self._setup_logger()

        self._labels_to_scores = {}
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

    @abc.abstractmethod
    def calculate_all_avg_scores(self):
        """
        Calculate the cosine similarity of the entire corpus
        :return: dictionary from user id to (average user score, label)
        """
        return {}

    @abc.abstractmethod
    def calculate_avg_score_for_group(self, group='control'):
        """
        Calculates the average score for a given group
        :param group: group name, either 'control' or 'patients'
        :return: average score (float)
        """
        return 0
