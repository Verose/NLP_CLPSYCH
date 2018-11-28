import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from word_embeddings.cosine_similarity.sliding_window import SlidingWindow
from word_embeddings.cosine_similarity.utils import get_vector_repr_of_word


class BasicSlidingWindow(SlidingWindow):
    def __init__(self, model, data, window_size, extras):
        super().__init__(model, data, window_size, extras)

    def calculate_all_avg_scores(self):
        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False):
            user_id = row[0]
            label = row[1]
            scores = []

            for answer in row[2:]:
                # some users didn't answer all of the questions
                if not answer or answer is pd.np.nan:
                    self._logger.debug('skipping empty answer for user: {}'.format(user_id))
                    continue
                answer = answer.split()
                # skip short answers
                if len(answer) < self._window_size + 1:
                    self._logger.debug('skipping short answer of length: {} for user: {}'.format(len(answer), user_id))
                    continue

                score = self._avg_answer_score_by_window(answer)
                scores += [score]

            avg_user_score = sum(scores) / len(scores)

            if label == 'control':
                self._control_scores += [(avg_user_score, user_id)]
            else:
                self._patient_scores += [(avg_user_score, user_id)]

    def calculate_avg_score_for_group(self, group='control'):
        if group == 'control':
            control_scores = [score[0] for score in self._control_scores]
            return sum(control_scores) / len(control_scores)
        else:
            patient_scores = [score[0] for score in self._patient_scores]
            return sum(patient_scores) / len(patient_scores)

    def _avg_answer_score_by_window(self, answer):
        """
        Calculate the cosine similarity of an answer using a sliding window
        :param answer: array of string representing an answer
        :return: cosine similarity score
        """
        scores = []

        for pos, word in enumerate(answer):
            if pos + self._window_size >= len(answer):
                break

            word_vector = get_vector_repr_of_word(self._model, word)
            score = 0

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = answer[pos + dist]
                context_vector = get_vector_repr_of_word(self._model, context)
                score += cosine_similarity([word_vector], [context_vector])[0][0]

            score /= self._window_size
            scores += [score]

        return sum(scores) / len(scores)
