import os

import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.metrics.pairwise import cosine_similarity

from common.utils import pos_tags_jsons_generator, DATA_DIR, OUTPUTS_DIR, \
    load_model, read_relevant_set, read_reference_set
from dependency_parser.generate_relevant_sets import repair_document
from dependency_parser.idf_scores import IdfScores


class DependencyCosSimScorer:
    def __init__(self):
        self._relevant_tags = {
            'control': {
                'noun': read_relevant_set('nouns', 'control'),
                'verb': read_relevant_set('verbs', 'control')
            },
            'patients': {
                'noun': read_relevant_set('nouns', 'patients'),
                'verb': read_relevant_set('verbs', 'patients')
            }
        }
        self._reference_tags = {
            'noun': read_reference_set('nouns'),
            'verb': read_reference_set('verbs')
        }

        # get part of speech tags
        self._answers_to_user_id_pos_data = {}
        pos_tags_generator = pos_tags_jsons_generator()
        for answer_num, ans_pos_tags in pos_tags_generator:
            self._answers_to_user_id_pos_data[answer_num] = ans_pos_tags

        # init model
        self._model = load_model('word2vec_dep.pickle')

        # calculate idf scores for words
        self._idf_scores = IdfScores(self.get_documents(), repair_document)
        self._idf_scores.calculate_idf_scores()

        self.missing_idf = {}
        self.words_without_embeddings = []
        self.missing_words = []
        self.pos_tags_used = {
            'control': {'nouns': [], 'verbs': []},
            'patients': {'nouns': [], 'verbs': []}
        }
        self.modifiers_used = {
            'control': {'noun': [], 'verb': []},
            'patients': {'noun': [], 'verb': []}
        }

    @staticmethod
    def get_documents():
        datasets = ['doctors_articles', '2b_healthy_articles', 'infomed_qa', 'haaretz_articles']
        files_grabbed = []
        root_dir = DATA_DIR

        for dir_name, _, file_list in os.walk(root_dir):
            if not any(dataset in dir_name for dataset in datasets):
                continue
            for file_name in file_list:
                full_path = os.path.join(dir_name, file_name)
                files_grabbed.append(full_path)
        return files_grabbed

    @staticmethod
    def score_group(avgs_nouns, avgs_verbs):
        avgs_nouns = np.mean([avg for avg in avgs_nouns if not np.isnan(avg)])
        avgs_verbs = np.mean([avg for avg in avgs_verbs if not np.isnan(avg)])
        return avgs_nouns, avgs_verbs

    def all_users_scores(self, users, users_scores):
        avgs_nouns = []
        avgs_verbs = []
        for user in users:
            avg_nouns, avg_verbs = self.score_user(user, users_scores)
            avgs_nouns += [avg_nouns]
            avgs_verbs += [avg_verbs]
        return avgs_nouns, avgs_verbs

    @staticmethod
    def score_user(user, users_scores):
        score_per_answer = users_scores[user]  # get scores for this user
        score_per_answer = [s for s in score_per_answer.values()]  # list of scores for each answer
        flat_nouns_scores = [item for sublist in score_per_answer for item in sublist[0]]
        flat_verbs_scores = [item for sublist in score_per_answer for item in sublist[1]]
        avg_nouns = np.mean(flat_nouns_scores)
        avg_verbs = np.mean(flat_verbs_scores)
        return avg_nouns, avg_verbs

    def cos_sim_scores_per_user(self, control, patients):
        def calc_mean(scores):
            return pd.np.nanmean(scores) if len(scores) > 0 else pd.np.nan

        users_scores = {}
        results = []

        for group, group_name in [control, patients]:
            for user in group:
                user_scores = self.cos_sim_score_for_user(user, group_name)
                users_scores[user] = user_scores
                [results.append(
                    (user, group_name, ans_num, calc_mean(ans_score[0]), calc_mean(ans_score[1])))
                    for ans_num, ans_score in user_scores.items()]

        columns = ["user_id", "label", "answer_num", "noun", "verb"]
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(os.path.join(DATA_DIR, "BY_incoherence_scores.csv"), index=False)

        return users_scores

    def cos_sim_score_for_user(self, user, group):
        scores = {}

        for answer_num, users_pos_data in sorted(self._answers_to_user_id_pos_data.items()):
            pos_data = users_pos_data[user]
            ans_noun_scores = []
            ans_verb_scores = []

            if not pos_data['lemmas']:
                scores[answer_num] = ([], [])
                continue
            pos_tags = pos_data['posTags']
            words = pos_data['lemmas']

            for word, pos_tag in zip(words, pos_tags):
                if pos_tag == 'noun' or pos_tag == 'verb':
                    score = self.cos_sim_for_tag(word, pos_tag, group)
                    if score and pos_tag == 'noun':
                        ans_noun_scores += [score]
                        self.pos_tags_used[group]['nouns'] += [word]
                    elif score and pos_tag == 'verb':
                        ans_verb_scores += [score]
                        self.pos_tags_used[group]['verbs'] += [word]

            scores[answer_num] = (ans_noun_scores, ans_verb_scores)
        return scores

    def cos_sim_for_tag(self, word, pos_tag, group):
        if word not in self._relevant_tags[group][pos_tag]:
            return None
        elif word not in self._reference_tags[pos_tag]:
            self.missing_words.append(word)
            return None

        relevant_context = self._relevant_tags[group][pos_tag][word]
        reference_context = self._reference_tags[pos_tag][word]
        score = self.score(relevant_context, reference_context, pos_tag, group)

        return score

    def score(self, relevant_contexts, reference_contexts, pos_tag, group):
        relevant_context_vector = np.zeros((300,))
        reference_context_vector = np.zeros((300,))

        relevant_context_vector = self.idf_score(
            relevant_context_vector,
            relevant_contexts,
            pos_tag=pos_tag,
            group=group,
            error_msg='relevant word')
        reference_context_vector = self.idf_score(
            reference_context_vector,
            reference_contexts,
            error_msg='reference word'
        )

        if not relevant_context_vector.any() or not reference_context_vector.any():
            return None

        cos_sim = cosine_similarity([relevant_context_vector], [reference_context_vector])[0][0]
        return cos_sim

    def idf_score(self, context_vector, contexts, pos_tag=None, group=None, error_msg=''):
        for context in contexts:
            idf = self._idf_scores.get_idf_score(context)
            if idf:
                if context in self._model:
                    context_vector += idf * self._model[context]
                    if group:
                        self.modifiers_used[group][pos_tag] += [context]
                else:
                    self.words_without_embeddings.append(context)
            else:
                self.missing_idf[context] = error_msg
        return context_vector

    @staticmethod
    def save_per_user_scores(avgs_nouns, avgs_verbs, users, group_name):
        dfs = []
        headers = ['id', 'nouns', 'verbs']

        for user, noun, verb in zip(users, avgs_nouns, avgs_verbs):
            df = pd.DataFrame([(user, noun, verb)], columns=headers)
            dfs += [df]

        dfs = pd.concat(dfs, axis=0)
        dfs.to_csv(os.path.join(OUTPUTS_DIR, "dependency_scores_{}.csv".format(group_name)), index=False)
        return dfs

    @staticmethod
    def ttest_group_scores(control_users_scores, patients_users_scores):
        control_users_scores = control_users_scores[['nouns', 'verbs']]
        patients_users_scores = patients_users_scores[['nouns', 'verbs']]
        ttest = stats.ttest_ind(control_users_scores, patients_users_scores)
        return ttest
