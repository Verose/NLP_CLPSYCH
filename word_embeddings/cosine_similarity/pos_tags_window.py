import glob
import json
import logging
import os
import string
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from word_embeddings.cosine_similarity.sliding_window import SlidingWindow
from word_embeddings.cosine_similarity.utils import get_vector_repr_of_word

logger = logging.getLogger('SlidingWindow')
logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class POSSlidingWindow(SlidingWindow):
    def __init__(self, model, data, window_size, data_dir):
        self._model = model
        self._data = data
        self._window_size = window_size
        self._data_dir = data_dir

        self._answers_to_user_id_pos_data = {}

        self._read_answers_pos_tags()

    def calculate_all_avg_scores(self):
        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False):
            user_id = row[0]
            label = row[1]

            scores = self._user_avg_scores(user_id)
            avg_user_score = sum(scores) / len(scores)
            self._labels_to_scores[user_id] = (avg_user_score, label)

            if label == 'control':
                self._control_scores += [(avg_user_score, user_id)]
            else:
                self._patient_scores += [(avg_user_score, user_id)]

        return self._labels_to_scores

    def calculate_avg_score_for_group(self, group='control'):
        if group == 'control':
            control_scores = [score[0] for score in self._control_scores]
            return sum(control_scores) / len(control_scores)
        else:
            patient_scores = [score[0] for score in self._patient_scores]
            return sum(patient_scores) / len(patient_scores)

    def _user_avg_scores(self, user_id):
        scores = []

        # iterate answers
        for answer_num, users_pos_data in self._answers_to_user_id_pos_data.items():
            user_pos_data = users_pos_data[user_id]

            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                logger.debug('skipping empty answer for user: {}'.format(user_id))
                continue

            score = self._avg_answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])
            scores += [score]

        return scores

    def _read_answers_pos_tags(self):
        json_pattern = os.path.join(self._data_dir, 'answers_pos_tags', '*.json')
        json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

        for file in json_files:
            with open(file, encoding='utf-8') as f:
                ans_pos_tags = json.load(f)
                self._answers_to_user_id_pos_data[os.path.basename(file).split('.')[0]] = ans_pos_tags

    def _avg_answer_score_by_pos_tags(self, answer, pos_tags):
        """
        Calculate the cosine similarity of an answer using POS tags
        Only considering “content words” - nouns, verbs, adjectives and adverbs
        :return:
        """
        scores = []
        valid_words = []
        valid_pos_tags = []

        for pos, (word, pos_tag) in enumerate(zip(answer, pos_tags)):
            if self._should_skip(word, pos_tag):
                continue
            valid_words += [word]
            valid_pos_tags += [pos_tag]

        if not valid_words:
            return 0

        for pos, (word, pos_tag) in enumerate(zip(valid_words, valid_pos_tags)):
            if pos + self._window_size >= len(valid_words):
                break

            word_vector = get_vector_repr_of_word(self._model, word, logger)
            score = 0

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = answer[pos + dist]
                context_vector = get_vector_repr_of_word(self._model, context, logger)
                score += cosine_similarity([word_vector], [context_vector])[0][0]

            score /= self._window_size
            scores += [score]

        return sum(scores) / len(scores) if len(scores) > 0 else 0

    @staticmethod
    def _should_skip(word, pos_tag):
        """
        Should skip words which are not noun/verb/adverb/adjective, and punctuation marks
        :param word: str
        :param pos_tag: str
        :return: True/False
        """
        if not (pos_tag == 'noun' or pos_tag == 'verb' or pos_tag == 'adverb' or pos_tag == 'adjective'):
            return True
        if word in string.punctuation:
            return True
        return False
