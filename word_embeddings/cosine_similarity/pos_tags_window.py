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
    def __init__(self, model, data, window_size, data_dir, pos_tags, window_method):
        self._data_dir = data_dir
        self._pos_tags_to_filter_in = 'noun verb adverb adjective' if pos_tags.lower() == 'content' else pos_tags
        self._window_method = window_method

        self._model = model
        self._data = data
        self._window_size = window_size
        self._logger = self._setup_logger()

        self._control_scores = []
        self._patients_scores = []
        self._control_users_to_question_scores = {}
        self._patients_users_to_question_scores = {}
        self._control_users_to_valid_words = {}
        self._patients_users_to_valid_words = {}

        self._answers_to_user_id_pos_data = {}
        self._control_repetitions = []
        self._patient_repetitions = []
        self._control_items = []
        self._patient_items = []

        self.words_without_embeddings = []
        self._read_answers_pos_tags()

    def calculate_all_scores(self):
        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False,
                               desc='Users'):
            user_id = row[0]
            label = row[1]

            averages = self._user_avg_scores(user_id)
            scores, repetitions, items, words = \
                averages['scores'], averages['repetitions'], averages['items'], averages['valid_words']
            avg_user_score = sum(scores.values()) / len(scores.values())
            avg_repetitions = sum(repetitions.values()) / len(repetitions.values())
            avg_items = sum(items.values()) / len(items.values())

            if label == 'control':
                self._control_users_to_question_scores[user_id] = scores
                self._control_users_to_valid_words[user_id] = words
                self._control_scores += [(avg_user_score, user_id)]
                self._control_repetitions += [(avg_repetitions, user_id)]
                self._control_items += [(avg_items, user_id)]
            else:
                self._patients_users_to_question_scores[user_id] = scores
                self._patients_users_to_valid_words[user_id] = words
                self._patients_scores += [(avg_user_score, user_id)]
                self._patient_repetitions += [(avg_repetitions, user_id)]
                self._patient_items += [(avg_items, user_id)]

    def calculate_group_scores(self, group='control'):
        scores = self.get_scores(group)
        return sum(scores) / len(scores)

    def calculate_repetitions_for_group(self, group='control'):
        if group == 'control':
            control_repetitions = [rep[0] for rep in self._control_repetitions]
            return sum(control_repetitions) / len(control_repetitions)
        else:
            patient_repetitions = [rep[0] for rep in self._patient_repetitions]
            return sum(patient_repetitions) / len(patient_repetitions)

    def calculate_items_for_group(self, group='control'):
        if group == 'control':
            control_items = [item[0] for item in self._control_items]
            return sum(control_items) / len(control_items)
        else:
            patient_items = [item[0] for item in self._patient_items]
            return sum(patient_items) / len(patient_items)

    def _user_avg_scores(self, user_id):
        skip = []
        scores = {}
        repetitions = {}
        items = {}
        valid_words = {}

        # iterate answers
        for answer_num, users_pos_data in sorted(self._answers_to_user_id_pos_data.items()):
            user_pos_data = users_pos_data[user_id]

            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                self._logger.debug('skipping empty answer for user: {}, setting with mean'.format(user_id))
                skip += [answer_num]
                continue

            ans_score = self._answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])

            if not ans_score:
                skip += [answer_num]
                continue

            scores[answer_num] = ans_score['scores']
            repetitions[answer_num] = ans_score['repetitions']
            items[answer_num] = ans_score['items']
            valid_words[answer_num] = ans_score['valid_words']

        # fill in for missing answers
        if skip:
            mean = sum(scores.values()) / len(scores.values())
            for ans in skip:
                scores[ans] = mean
                repetitions[ans] = 0
                valid_words[ans] = []

        return {'scores': scores, 'repetitions': repetitions, 'items': items, 'valid_words': valid_words}

    def _read_answers_pos_tags(self):
        pos_tags_generator = pos_tags_jsons_generator()

        for answer_num, ans_pos_tags in pos_tags_generator:
            self._answers_to_user_id_pos_data[answer_num] = ans_pos_tags

    def _answer_score_by_pos_tags(self, answer, pos_tags):
        """
        Calculate:
        scores - average/min/max cosine similarity of an answer using POS tags (average of window averages)
        Cosine Similarity is calculated as an average over the answers.
        An answer is calculated by the average cosine similarity over all possible windows
        repetitions - sum of word repetitions for an answer
        Word repetitions are calculated as an average over the answers.
        An answer is calculated by summation over all possible windows.
        items - sum of valid words for an answer
        Only considering “content words” - nouns, verbs, adjectives and adverbs
        'all' considers all possible pos tags.
        :return: dictionary with the required fields
        """
        valid_words = []
        valid_pos_tags = []

        # get a list of valid words and their pos_tags
        for i, (word, pos_tag) in enumerate(zip(answer, pos_tags)):
            if self._should_skip(word, pos_tag):
                continue
            valid_words += [word]
            valid_pos_tags += [pos_tag]

        if not valid_words:
            return None

        if self._window_method == 'forward':
            repetitions, scores = self._answer_scores_forward_window(valid_words)
        else:
            repetitions, scores = self._avg_answer_scores_surrounding_window(valid_words)

        ans_scores = None
        if scores:
            ret_score = sum(scores) / len(scores)
            ans_scores = {'scores': ret_score, 'repetitions': repetitions, 'items': len(valid_words),
                          'valid_words': valid_words}
        return ans_scores

    def _answer_scores_forward_window(self, valid_words):
        scores = []
        repetitions = 0

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

                if word == context:
                    repetitions += 1

            # average for window
            score = np.mean(win_scores)
            scores += [score]
        return repetitions, scores

    def _avg_answer_scores_surrounding_window(self, valid_words):
        scores = []
        repetitions = 0

        for i, word in enumerate(valid_words):
            if i - self._window_size < 0:
                continue
            if i + self._window_size >= len(valid_words):
                break

            word_vector = self._model[word]
            score = 0

            # calculate cosine similarity for surrounding window
            for dist in range(i - self._window_size, i + self._window_size + 1):
                if dist == i:
                    continue

                context = valid_words[dist]
                context_vector = self._model[context]
                score += cosine_similarity([word_vector], [context_vector])[0][0]

                if word == context:
                    repetitions += 1

            # average for window
            score /= (2 * self._window_size)
            scores += [score]
        return repetitions, scores

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

    def get_scores(self, group='control'):
        group = self._control_scores if group == 'control' else self._patients_scores
        return [score[0] for score in group]

    def get_user_to_question_scores(self, group='control'):
        return self._control_users_to_question_scores if group == 'control' \
            else self._patients_users_to_question_scores

    def get_user_to_question_valid_words(self, group='control'):
        return self._control_users_to_valid_words if group == 'control' else self._patients_users_to_valid_words

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
