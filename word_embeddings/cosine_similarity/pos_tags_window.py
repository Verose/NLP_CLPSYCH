import json
import os
import string
from multiprocessing import Pool, Value

import numpy as np
from scipy.stats import stats
from sklearn.metrics.pairwise import cosine_similarity

from word_embeddings.common.utils import pos_tags_jsons_generator, get_vector_for_word, get_words_in_model


def init(args):
    """ store the counter for later use """
    global counter
    counter = args


class POSSlidingWindow:
    def __init__(self, model, data, window_size, data_dir, pos_tags, questions, question_minimum_length, is_rsdd=False):
        self._data_dir = data_dir
        self._model = model

        if not is_rsdd:
            self._pos_tags_to_filter_in = \
                'noun verb adverb adjective'.split() if pos_tags.lower() == 'content' else pos_tags.lower().split()
        else:
            self._pos_tags_to_filter_in = \
                'NN NNP NNPS NNS' \
                'JJ JJR JJS'  \
                'VB VBD VBG VBN VBP VBZ' \
                'RB RBR RBS RP'.split() if pos_tags.lower() == 'content' else pos_tags.lower().split()

        self._data = data.to_dict('records')
        self._window_size = window_size
        self._questions = questions
        self._question_minimum_length = question_minimum_length
        self._is_rsdd = is_rsdd
        self._total = len(self._data)

        self._control_users_to_agg_score = {}
        self._patients_users_to_agg_score = {}

        self._answers_to_user_id_pos_data = {}

    def calculate_all_scores(self):
        # iterate users
        pool = Pool(processes=8, initializer=init, initargs=(Value('i', 0), ))
        results = pool.map(self._calc_scores_per_user, self._data)
        pool.close()
        pool.join()

        for user_id, label, score in results:
            self._update_users_data(user_id, label, score)

    def _calc_scores_per_user(self, data):
        user_id = data['id']
        label = data['label']

        score = self._user_avg_score(user_id)
        global counter
        with counter.get_lock():
            counter.value += 1

        if counter.value % 2 == 0:
            print("finished {}/{}".format(counter.value, self._total))

        return user_id, label, score

    def _update_users_data(self, user_id, label, score):
        if label == 'control':
            self._control_users_to_agg_score[user_id] = score
        else:
            self._patients_users_to_agg_score[user_id] = score

    def calculate_group_scores(self, group='control'):
        scores = self.get_avg_scores(group)
        return np.mean(scores)

    def _user_avg_score(self, user_id):
        sum_scores = 0
        scores_counter = 0

        # iterate answers
        for answer_num, user_pos_data in self._answers_pos_tags_generator(user_id):
            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                continue

            ans_score = self._answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])

            if not ans_score:
                continue

            sum_scores += ans_score
            scores_counter += 1

        return sum_scores/scores_counter if scores_counter > 0 else np.nan

    def _read_answers_pos_tags(self):
        pos_tags_generator = pos_tags_jsons_generator()

        for answer_num, ans_pos_tags in pos_tags_generator:
            if self._questions is not None and answer_num not in self._questions:
                continue
            self._answers_to_user_id_pos_data[answer_num] = ans_pos_tags

        return len(self._answers_to_user_id_pos_data) > 0

    def _answers_pos_tags_generator(self, user_id):
        """
        Generator for pos tags.
        :param user_id: user id
        For the hebrew dataset, the pos tags data is read from json files.
        For the english dataset, the pos tags are calculated from the posts.
        :return: (answer number, dictionary with 'tokens' list, and 'posTags' list)
        """
        if not self._is_rsdd:
            if len(self._answers_to_user_id_pos_data) == 0:
                assert self._read_answers_pos_tags(), "No pos tags found!"

            for answer_num, users_pos_data in sorted(self._answers_to_user_id_pos_data.items()):
                user_pos_data = users_pos_data[user_id]
                yield answer_num, user_pos_data
        else:
            with open(os.path.join(self._data_dir, 'pos_tags_rsdd', '{}.json'.format(user_id)), encoding='utf-8') as f:
                ans_pos_tags = json.load(f)
                tokens_list = ans_pos_tags['tokens']
                pos_tags_list = ans_pos_tags['posTags']

                for answer_num, (tokens, pos_tags) in enumerate(zip(tokens_list, pos_tags_list), 1):
                    if answer_num > 500:  # TODO hard coded limit for posts
                        break

                    yield answer_num, {'tokens': tokens, 'posTags': pos_tags}

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
        previous_valid_word = ''
        words_in_model_dict = get_words_in_model(self._model, answer)

        # get a list of valid words and their pos_tags
        for i, (word, pos_tag) in enumerate(zip(answer, pos_tags)):
            if not words_in_model_dict[word]:
                continue
            if self._should_skip(word, pos_tag):
                continue
            # skip duplicate words
            if word == previous_valid_word:
                continue
            valid_words += [word]
            previous_valid_word = word

        if not valid_words:
            return None
        # skip sentences too short
        if len(valid_words) < self._question_minimum_length:
            return None

        scores = self._answer_scores_forward_window(valid_words)

        ans_avg_scores = None
        if scores:
            ans_avg_scores = sum(scores) / len(scores)
        return ans_avg_scores

    def _answer_scores_forward_window(self, valid_words):
        scores = []
        num_vectors = len(valid_words)
        valid_vectors = get_vector_for_word(self._model, valid_words)

        for i, word_vector in enumerate(valid_vectors):
            if i + self._window_size >= num_vectors:
                break

            win_scores = []

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context_vector = valid_vectors[i + dist]
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
        if pos_tag not in self._pos_tags_to_filter_in:
            if 'all' in self._pos_tags_to_filter_in:
                return False
            return True
        return False

    def get_avg_scores(self, group='control'):
        group = self._control_users_to_agg_score if group == 'control' else self._patients_users_to_agg_score
        avg_scores = [score for score in group.values() if not np.isnan(score)]
        return avg_scores

    def perform_ttest_on_averages(self):
        """
        Performs t-test on the average cos-sim score of each user
        :return:
        """
        control_scores = self.get_avg_scores('control')
        patient_scores = self.get_avg_scores('patients')
        ttest = stats.ttest_ind(control_scores, patient_scores)
        return ttest
