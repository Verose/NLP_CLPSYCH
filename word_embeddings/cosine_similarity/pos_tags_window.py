import string
import sys

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from word_embeddings.cosine_similarity.sliding_window import SlidingWindow
from word_embeddings.cosine_similarity.utils import get_vector_repr_of_word, pos_tags_jsons_generator


class POSSlidingWindow(SlidingWindow):
    def __init__(self, model, data, window_size, data_dir, pos_tags, window_method):
        self._data_dir = data_dir
        self._pos_tags_to_filter_in = pos_tags
        self._window_method = window_method

        self._answers_to_user_id_pos_data = {}
        self._control_repetitions = []
        self._patient_repetitions = []
        self._control_items = []
        self._patient_items = []

        self._read_answers_pos_tags()
        super().__init__(model, data, window_size)

    def calculate_all_avg_scores(self):
        # iterate users
        for index, row in tqdm(self._data.iterrows(), file=sys.stdout, total=len(self._data), leave=False):
            user_id = row[0]
            label = row[1]

            averages = self._user_avg_scores(user_id)
            scores, repetitions, items = averages['scores'], averages['repetitions'], averages['items']
            avg_user_score = sum(scores.values()) / len(scores.values())
            avg_repetitions = sum(repetitions.values()) / len(repetitions.values())
            avg_items = sum(items.values()) / len(items.values())

            if label == 'control':
                self._control_users_to_question_scores[user_id] = scores
                self._control_scores += [(avg_user_score, user_id)]
                self._control_repetitions += [(avg_repetitions, user_id)]
                self._control_items += [(avg_items, user_id)]
            else:
                self._patient_users_to_question_scores[user_id] = scores
                self._patient_scores += [(avg_user_score, user_id)]
                self._patient_repetitions += [(avg_repetitions, user_id)]
                self._patient_items += [(avg_items, user_id)]

    def calculate_avg_score_for_group(self, group='control'):
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

        # iterate answers
        for answer_num, users_pos_data in sorted(self._answers_to_user_id_pos_data.items()):
            user_pos_data = users_pos_data[user_id]

            # some users didn't answer all of the questions
            if not user_pos_data['tokens']:
                self._logger.debug('skipping empty answer for user: {}, setting with mean'.format(user_id))
                skip += [answer_num]
                continue

            averages = self._avg_answer_score_by_pos_tags(user_pos_data['tokens'], user_pos_data['posTags'])
            scores[answer_num] = averages['scores']
            repetitions[answer_num] = averages['repetitions']
            items[answer_num] = averages['items']

        # fill in for missing answers
        if skip:
            mean = sum(scores.values()) / len(scores.values())
            for ans in skip:
                scores[ans] = mean
                repetitions[ans] = 0

        return {'scores': scores, 'repetitions': repetitions, 'items': items}

    def _read_answers_pos_tags(self):
        pos_tags_generator = pos_tags_jsons_generator(self._data_dir)

        for answer_num, ans_pos_tags in pos_tags_generator:
            self._answers_to_user_id_pos_data[answer_num] = ans_pos_tags

    def _avg_answer_score_by_pos_tags(self, answer, pos_tags):
        """
        Calculate:
        scores - average cosine similarity of an answer using POS tags (average of window averages)
        Cosine Similarity is calculated as an average over the answers.
        An answer is calculated by the average cosine similarity over all possible windows
        repetitions - sum of word repetitions for an answer
        Word repetitions are calculated as an average over the answers.
        An answer is calculated by summation over all possible windows.
        items - sum of valid words for an answer
        Only considering “content words” - nouns, verbs, adjectives and adverbs
        :return: dictionary with the required fields
        """
        valid_words = []
        valid_pos_tags = []

        for i, (word, pos_tag) in enumerate(zip(answer, pos_tags)):
            if self._should_skip(word, pos_tag):
                continue
            valid_words += [word]
            valid_pos_tags += [pos_tag]

        if not valid_words:
            return {'scores': 0, 'repetitions': 0, 'items': 0}

        if self._window_method == 'forward':
            repetitions, scores = self._avg_answer_scores_forward_window(valid_pos_tags, valid_words)
        else:
            repetitions, scores = self._avg_answer_scores_surrounding_window(valid_pos_tags, valid_words)

        ret_score = sum(scores) / len(scores) if len(scores) > 0 else 0
        return {'scores': ret_score, 'repetitions': repetitions, 'items': len(valid_words)}

    def _avg_answer_scores_forward_window(self, valid_pos_tags, valid_words):
        scores = []
        repetitions = 0

        for i, (word, pos_tag) in enumerate(zip(valid_words, valid_pos_tags)):
            if i + self._window_size >= len(valid_words):
                break

            word_vector = get_vector_repr_of_word(self._model, word)
            score = 0

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = valid_words[i + dist]
                context_vector = get_vector_repr_of_word(self._model, context)
                score += cosine_similarity([word_vector], [context_vector])[0][0]

                if word == context:
                    repetitions += 1

            # average for window
            score /= self._window_size
            scores += [score]
        return repetitions, scores

    def _avg_answer_scores_surrounding_window(self, valid_pos_tags, valid_words):
        scores = []
        repetitions = 0

        for i, (word, pos_tag) in enumerate(zip(valid_words, valid_pos_tags)):
            if i - self._window_size < 0:
                continue
            if i + self._window_size >= len(valid_words):
                break

            word_vector = get_vector_repr_of_word(self._model, word)
            score = 0

            # calculate cosine similarity for surrounding window
            for dist in range(i - self._window_size, i + self._window_size + 1):
                if dist == i:
                    continue

                context = valid_words[dist]
                context_vector = get_vector_repr_of_word(self._model, context)
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
        :param word: str
        :param pos_tag: str
        :return: True/False
        """
        if pos_tag not in self._pos_tags_to_filter_in:
            return True
        if word in string.punctuation:
            return True
        return False
