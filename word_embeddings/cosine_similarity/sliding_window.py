import abc


class SlidingWindow(object):
    __metaclass__ = abc.ABCMeta
    _labels_to_scores = {}
    _control_scores = []
    _patient_scores = []

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
