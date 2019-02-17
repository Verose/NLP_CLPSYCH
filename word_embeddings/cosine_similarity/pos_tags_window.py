import string
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from word_embeddings.cosine_similarity.sliding_window import SlidingWindow
from word_embeddings.common.utils import pos_tags_jsons_generator


class POSSlidingWindow(SlidingWindow):
    def __init__(self, model, data, window_size, data_dir, pos_tags, window_method):
        self._data_dir = data_dir
        self._pos_tags_to_filter_in = 'noun verb adverb adjective' if pos_tags.lower() == 'content' else pos_tags
        self._window_method = window_method

        self._answers_to_user_id_pos_data = {}
        self._control_repetitions = []
        self._patient_repetitions = []
        self._control_items = []
        self._patient_items = []

        self.words_without_embeddings = []
        self._read_answers_pos_tags()
        super().__init__(model, data, window_size)

    def calculate_all_scores(self):
        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False,
                               desc='Users'):
            user_id = row[0]
            label = row[1]

            user_scores = self._user_scores(user_id)
            scores, repetitions, items, words = \
                user_scores['scores'], user_scores['repetitions'], user_scores['items'], user_scores['valid_words']
            avg_user_score = np.mean([val['mean'] for val in scores.values()])
            min_user_score = np.min([val['min'] for val in scores.values()])
            max_user_score = np.max([val['max'] for val in scores.values()])
            avg_repetitions = sum(repetitions.values()) / len(repetitions.values())
            avg_items = sum(items.values()) / len(items.values())

            if label == 'control':
                self._control_users_to_question_scores[user_id] = scores
                self._control_users_to_valid_words[user_id] = words
                self._control_scores['mean'] += [(avg_user_score, user_id)]
                self._control_scores['min'] += [(min_user_score, user_id)]
                self._control_scores['max'] += [(max_user_score, user_id)]
                self._control_repetitions += [(avg_repetitions, user_id)]
                self._control_items += [(avg_items, user_id)]
            else:
                self._patients_users_to_question_scores[user_id] = scores
                self._patients_users_to_valid_words[user_id] = words
                self._patients_scores['mean'] += [(avg_user_score, user_id)]
                self._patients_scores['min'] += [(min_user_score, user_id)]
                self._patients_scores['max'] += [(max_user_score, user_id)]
                self._patient_repetitions += [(avg_repetitions, user_id)]
                self._patient_items += [(avg_items, user_id)]

    def calculate_group_scores(self, group='control'):
        mean_scores, min_scores, max_scores = self.get_scores(group)
        return np.mean(mean_scores), np.min(min_scores), np.max(max_scores)

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

    def _user_scores(self, user_id):
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

            ans_scores = self._answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])

            if not ans_scores:
                skip += [answer_num]
                continue

            scores[answer_num] = ans_scores['scores']
            repetitions[answer_num] = ans_scores['repetitions']
            items[answer_num] = ans_scores['items']
            valid_words[answer_num] = ans_scores['valid_words']

        # fill in for missing/empty answers/scores
        if skip:
            # set to averages so it doesn't affect the scores
            for ans in skip:
                # if scores:
                #     scores[ans] = {
                #         'mean': np.mean([val['mean'] for val in scores.values()]),
                #         'min': np.mean([val['min'] for val in scores.values()]),
                #         'max': np.mean([val['max'] for val in scores.values()])
                #     }
                # else:
                #     scores[ans] = {'mean': 0, 'min': 0, 'max': 0}
                scores[ans] = {
                    'mean': np.mean([val['mean'] for val in scores.values()]),
                    'min': np.mean([val['min'] for val in scores.values()]),
                    'max': np.mean([val['max'] for val in scores.values()])
                }
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
            ret_score = {
                'mean': np.mean(scores['mean']),
                'min': np.min(scores['min']),
                'max': np.max(scores['max'])
            }
            ans_scores = {'scores': ret_score, 'repetitions': repetitions, 'items': len(valid_words),
                          'valid_words': valid_words}
        return ans_scores

    def _answer_scores_forward_window(self, valid_words):
        scores = defaultdict(list)
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

            # average/min/max scores for window
            scores['mean'] += [np.mean(win_scores)]
            scores['min'] += [np.min(win_scores)]
            scores['max'] += [np.max(win_scores)]
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
