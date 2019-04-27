import logging
import os
import string
import sys
from datetime import datetime
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

import numpy as np
from scipy.stats import stats
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from word_embeddings.common.utils import pos_tags_jsons_generator


class POSSlidingWindow:
    def __init__(self, model, data, window_size, data_dir, pos_tags, questions, question_minimum_length):
        self._data_dir = data_dir
        self._pos_tags_to_filter_in = 'noun verb adverb adjective' if pos_tags.lower() == 'content' else pos_tags

        self._model = model
        self._data = data
        self._window_size = window_size
        self._questions = questions
        self._question_minimum_length = question_minimum_length
        self._logger = self._setup_logger()

        self._control_users_to_avg_scores = {}
        self._patients_users_to_avg_scores = {}
        self._control_users_to_valid_words = {}
        self._patients_users_to_valid_words = {}

        self._answers_to_user_id_pos_data = {}

        self.words_without_embeddings = []
        assert self._read_answers_pos_tags(), "No pos tags found!"

    def calculate_all_scores(self):
        # iterate users
        for _, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False, desc='Users'):
            user_id = row[0]
            label = row[1]

            averages = self._user_avg_scores(user_id)
            scores, words = averages['avg_scores'], averages['valid_words']

            if label == 'control':
                self._control_users_to_avg_scores[user_id] = scores
                self._control_users_to_valid_words[user_id] = words
            else:
                self._patients_users_to_avg_scores[user_id] = scores
                self._patients_users_to_valid_words[user_id] = words

    def calculate_group_scores(self, group='control'):
        scores = self.get_avg_scores(group)
        return np.mean(scores)

    def _user_avg_scores(self, user_id):
        avg_scores = {}
        valid_words = {}

        # iterate answers
        for answer_num, users_pos_data in sorted(self._answers_to_user_id_pos_data.items()):
            user_pos_data = users_pos_data[user_id]

            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                self._logger.debug('skipping empty answer for user: {}'.format(user_id))
                avg_scores[answer_num] = -2
                continue

            ans_score = self._answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])

            if not ans_score:
                avg_scores[answer_num] = -2
                continue

            avg_scores[answer_num] = ans_score['avg_scores']
            valid_words[answer_num] = ans_score['valid_words']

        return {'avg_scores': avg_scores, 'valid_words': valid_words}

    def _read_answers_pos_tags(self):
        pos_tags_generator = pos_tags_jsons_generator()

        for answer_num, ans_pos_tags in pos_tags_generator:
            if answer_num not in self._questions:
                continue
            self._answers_to_user_id_pos_data[answer_num] = ans_pos_tags

        return len(self._answers_to_user_id_pos_data) > 0

    def _answer_score_by_pos_tags(self, answer, pos_tags):
        """
        Calculate:
        scores - average cosine similarity of an answer using POS tags (average of window averages)
        Cosine Similarity is calculated as an average over the answers.
        An answer is calculated by the average cosine similarity over all possible windows
        An answer is calculated by summation over all possible windows.
        Only considering “content words” - nouns, verbs, adjectives and adverbs
        'all' considers all possible pos tags.
        :return: dictionary with the required fields
        """
        valid_words = []
        valid_pos_tags = []
        previous_valid_word = ''

        # get a list of valid words and their pos_tags
        for i, (word, pos_tag) in enumerate(zip(answer, pos_tags)):
            if self._should_skip(word, pos_tag):
                continue
            # skip duplicate words
            if word == previous_valid_word:
                continue
            valid_words += [word]
            valid_pos_tags += [pos_tag]
            previous_valid_word = word

        if not valid_words:
            return None
        # skip sentences too short
        if len(valid_words) < self._question_minimum_length:
            return None

        scores = self._answer_scores_forward_window(valid_words)

        ans_avg_scores = None
        if scores:
            ret_score = sum(scores) / len(scores)
            ans_avg_scores = {'avg_scores': ret_score, 'valid_words': valid_words}
        return ans_avg_scores

    def _answer_scores_forward_window(self, valid_words):
        scores = []

        for i, word in enumerate(valid_words):
            if i + self._window_size >= len(valid_words):
                break

            word_vector = self._model[word]
            win_scores = []

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = valid_words[i + dist]
                context_vector = self._model[context]
                win_scores.append(cosine_similarity([word_vector], [context_vector])[0][0])

            # average for window
            score = np.mean(win_scores)
            scores += [score]
        return scores

    def _should_skip(self, word, pos_tag):
        """
        Should skip specific pos tags (e.g. noun/verb/adverb/adjective), and punctuation marks
        'all' considers all possible pos tags.
        :param word: str
        :param pos_tag: str
        :return: True/False
        """
        if word in string.punctuation:
            return True
        if word not in self._model:
            self.words_without_embeddings.append(word)
            return True
        if pos_tag not in self._pos_tags_to_filter_in:
            if self._pos_tags_to_filter_in.lower() == 'all':
                return False
            return True
        return False

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

    def get_user_to_avg_scores(self, group='control'):
        group = self._control_users_to_avg_scores if group == 'control' else self._patients_users_to_avg_scores
        averages = {}
        for user, user_scores in group.items():
            averages[user] = np.mean([score for score in user_scores.values() if score > -2])
        return averages

    def get_avg_scores(self, group='control'):
        group = self._control_users_to_avg_scores if group == 'control' else self._patients_users_to_avg_scores
        averages = [np.mean([score for score in user_scores.values() if score > -2]) for user_scores in group.values()]
        averages = [avg for avg in averages if not np.isnan(avg)]
        return averages

    def get_user_to_question_scores(self, group='control'):
        return self._control_users_to_avg_scores if group == 'control' else self._patients_users_to_avg_scores

    def get_user_to_question_valid_words(self, group='control'):
        return self._control_users_to_valid_words if group == 'control' else self._patients_users_to_valid_words

    def perform_ttest_on_averages(self):
        """
        Performs t-test on the average cos-sim score of each user
        :return:
        """
        control_scores = self.get_avg_scores('control')
        patient_scores = self.get_avg_scores('patients')
        ttest = stats.ttest_ind(control_scores, patient_scores)
        return ttest

    def perform_ttest_on_all(self):
        """
        Performs t-test on all of the cos-sim scores of each user
        :return:
        """
        control_scores_by_question = self.get_user_to_question_scores('control')
        patient_scores_by_question = self.get_user_to_question_scores('patients')

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
