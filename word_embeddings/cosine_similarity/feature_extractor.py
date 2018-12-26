import glob
import json
import os
import sys

from polyglot.text import Text
from tqdm import tqdm

from word_embeddings.cosine_similarity.classifier_records import PosData


class FeatureExtractor:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._users = None
        self._answers_to_user_id_pos_tags = {}
        self._qnum_to_user_id_pos_features = {}

    def read_answers_pos_tags(self):
        json_pattern = os.path.join(self._data_dir, 'answers_pos_tags', '*.json')
        json_files = [pos_json for pos_json in glob.glob(json_pattern) if pos_json.endswith('.json')]

        for file in json_files:
            with open(file, encoding='utf-8') as f:
                ans_pos_tags = json.load(f)
                self._answers_to_user_id_pos_tags[int(os.path.basename(file).split('.')[0])] = ans_pos_tags
        self._users = list(self._answers_to_user_id_pos_tags[1].keys())

    def update_pos_tags_data(self):
        for user_id in tqdm(self._users, file=sys.stdout, total=len(self._users), leave=False):
            ans_pos_data = {}
            skipped = []

            for qnum, users_pos_data in sorted(self._answers_to_user_id_pos_tags.items()):
                user_pos_data = users_pos_data[user_id]['posTags']

                if not user_pos_data:
                    skipped += [user_id]
                    break

                nouns, verbs, adverbs, adjectives, sum_all = 0, 0, 0, 0, 0
                for pos_tag in user_pos_data:
                    if pos_tag == 'punctuation':
                        continue
                    if pos_tag == 'noun':
                        sum_all += 1
                        nouns += 1
                    elif pos_tag == 'verb':
                        sum_all += 1
                        verbs += 1
                    elif pos_tag == 'adverb':
                        sum_all += 1
                        adverbs += 1
                    elif pos_tag == 'adjective':
                        sum_all += 1
                        adjectives += 1
                ans_pos_data[qnum] = PosData(
                    user_id,
                    qnum,
                    nouns/sum_all if sum_all > 0 else 0,
                    verbs/sum_all if sum_all > 0 else 0,
                    adjectives/sum_all if sum_all > 0 else 0,
                    adverbs/sum_all if sum_all > 0 else 0
                )
            if user_id not in skipped:
                self._qnum_to_user_id_pos_features[user_id] = ans_pos_data

    def update_cosine_sim_scores(self, control_scores_by_question, patient_scores_by_question):
        for user_id, scores in control_scores_by_question.items():
            if user_id in self._qnum_to_user_id_pos_features:
                for qnum, score in scores.items():
                    self._qnum_to_user_id_pos_features[user_id][qnum].cossim_score = score
        for user_id, scores in patient_scores_by_question.items():
            if user_id in self._qnum_to_user_id_pos_features:
                for qnum, score in scores.items():
                    self._qnum_to_user_id_pos_features[user_id][qnum].cossim_score = score

    def update_sentiment_scores(self):
        for user_id in tqdm(self._users, file=sys.stdout, total=len(self._users), leave=False):
            if user_id not in self._qnum_to_user_id_pos_features:
                continue

            for qnum, users_pos_data in sorted(self._answers_to_user_id_pos_tags.items()):
                user_tokens = users_pos_data[user_id]['tokens']

                if not user_tokens:
                    break

                answer = ' '.join(user_tokens)
                text = Text(answer, 'he')
                self._qnum_to_user_id_pos_features[user_id][qnum].sentiment = text.polarity

    def get_features(self):
        features = [feat for feat in self._qnum_to_user_id_pos_features.values()]
        features = [list(q.values()) for q in features]
        features = [feat for sublist in features for feat in sublist]
        return features
