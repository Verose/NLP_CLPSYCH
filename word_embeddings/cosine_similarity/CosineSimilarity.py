import logging
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger('CosineSimilarity')
logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


class CosineSimilarity:
    def __init__(self, model, data, window_size):
        self._model = model
        self._data = data
        self._window_size = window_size
        self._labels_to_scores = {}
        self._control_scores = []
        self._patient_scores = []

    def calculate_all_avg_scores(self):
        """
        Calculate the cosine similarity of the entire corpus
        :return:
        """

        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data)):
            user_id = row[0]
            label = row[1]
            scores = []

            for answer in row[2:]:
                # some users didn't answer all of the questions
                if not answer or answer is pd.np.nan:
                    continue
                answer = answer.split()
                # skip short answers
                if len(answer) < self._window_size + 1:
                    continue

                score = self.avg_answer_score(answer)
                scores += [score]

            avg_user_score = sum(scores) / len(scores)
            self._labels_to_scores[user_id] = (avg_user_score, label)

            if label == 'control':
                self._control_scores += [(avg_user_score, user_id)]
            else:
                self._patient_scores += [(avg_user_score, user_id)]

        logger.info('success! finished')
        return self._labels_to_scores

    def calculate_avg_score_for_group(self, group='control'):
        if group == 'control':
            control_scores = [score[0] for score in self._control_scores]
            return sum(control_scores) / len(control_scores)
        else:
            patient_scores = [score[0] for score in self._patient_scores]
            return sum(patient_scores) / len(patient_scores)

    def _get_vector_repr_of_word(self, word):
        try:
            return self._model[word]
        except KeyError:
            if str.isdecimal(word):
                replacement_word = '<מספר>'
            elif str.isalpha(word):
                replacement_word = '<אנגלית>'
            elif any(i.isdigit() for i in word) and any("\u0590" <= c <= "\u05EA" for c in word):
                replacement_word = '<אות ומספר>'
            else:
                replacement_word = '<לא ידוע>'
            logger.info('word: {} replaced with: {}'.format(word, replacement_word))
            return self._model[replacement_word]

    def avg_answer_score(self, answer):
        """
        Calculate the cosine similarity of an answer
        :param answer: array of string representing an answer
        :return: cosine similarity score
        """

        scores = []

        for pos, word in enumerate(answer):
            if pos + self._window_size >= len(answer):
                break

            word_vector = self._get_vector_repr_of_word(word)
            score = 0

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = answer[pos + dist]
                context_vector = self._get_vector_repr_of_word(context)
                score += cosine_similarity([word_vector], [context_vector])[0][0]

            score /= self._window_size
            scores += [score]

        return sum(scores) / len(scores)
