import os
import string

import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from common.utils import get_vector_for_word, get_words_in_model, load_model


def init(args):
    """ store the counter for later use """
    global counter
    counter = args


class POSSlidingWindow:
    def __init__(self, data, window_size, data_dir, pos_tags, run_params, answers_pos_tags_generator,
                 pos_tags_type='explicit'):
        self._data_dir = data_dir
        self._model = None

        if pos_tags_type == 'explicit':
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
        self._question_minimum_length = run_params['question_minimum_length']
        self._embeddings_path = run_params['word_embeddings']
        self._total = len(self._data)

        self._control_users_to_agg_score = {}
        self._patients_users_to_agg_score = {}
        self._answers_pos_tags_generator = answers_pos_tags_generator
        self.derailment_precalc_scores = os.path.join(
            self._data_dir,
            "{}_win_{}_{}_scores.csv".format(run_params['dataset_name'], self._window_size, pos_tags)
        )

    def calculate_all_scores(self):
        if os.path.isfile(self.derailment_precalc_scores):
            results_df = pd.read_csv(self.derailment_precalc_scores)
        else:
            self._model = load_model(self._embeddings_path)
            # iterate users
            results = []

            for user_data in tqdm(self._data, total=len(self._data), desc="Creating scores csv file"):
                result = self._calc_scores_per_user(user_data)
                results.extend(result)

            columns = ["user_id", "label", "answer_num", "valid_words_cnt", "score"]
            results_df = pd.DataFrame(results, columns=columns)
            results_df.to_csv(self.derailment_precalc_scores, index=False)

        self._update_users_data(results_df)

    def _update_users_data(self, results_df):
        control_rows = results_df[
            (results_df['label'] == 'control') & (results_df['valid_words_cnt'] >= self._question_minimum_length)]
        patients_rows = results_df[
            (results_df['label'] != 'control') & (results_df['valid_words_cnt'] >= self._question_minimum_length)]
        control_scores = control_rows.groupby(['user_id'])['score'].mean()
        patient_scores = patients_rows.groupby(['user_id'])['score'].mean()

        for user_id, score in control_scores.iteritems():
            self._control_users_to_agg_score[user_id] = score
        for user_id, score in patient_scores.iteritems():
            self._patients_users_to_agg_score[user_id] = score

    def calculate_group_scores(self, group='control'):
        scores = self.get_avg_scores(group)
        return np.mean(scores)

    def _calc_scores_per_user(self, data):
        user_id = data['id']
        label = data['label']
        answer_data = []

        # iterate answers
        for answer_num, user_pos_data in self._answers_pos_tags_generator(user_id):
            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                answer_data.append((user_id, label, answer_num, 0, np.nan))
                continue
            ans_score, num_valid_words = self._answer_score_by_pos_tags(
                user_pos_data['tokens'], user_pos_data['posTags'])
            answer_data.append((user_id, label, answer_num, num_valid_words, ans_score))
        return answer_data

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
            return None, 0

        scores = self._answer_scores_forward_window(valid_words)

        ans_avg_score = None
        if scores:
            ans_avg_score = sum(scores) / len(scores)
        return ans_avg_score, len(valid_words)

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
