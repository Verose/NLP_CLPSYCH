import glob
import os

import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

from utils import remove_females, remove_depressed, DATA_DIR
from word_embeddings.cosine_similarity.utils import get_vector_repr_of_word, pos_tags_jsons_generator
from word_embeddings.dependency_parser.generate_relevant_sets import read_relevant_set, read_reference_set, \
    repair_document
from word_embeddings.dependency_parser.idf_scores import IdfScores


class DependencyCosSimScorer:
    def __init__(self):
        self.relevant_tags = {
            'control': {
                'noun': read_relevant_set('nouns', 'control'),
                'verb': read_relevant_set('verbs', 'control')
            },
            'patients': {
                'noun': read_relevant_set('nouns', 'patients'),
                'verb': read_relevant_set('verbs', 'patients')
            }
        }
        self.reference_tags = {
            'noun': read_reference_set('nouns'),
            'verb': read_reference_set('verbs')
        }

        # get part of speech tags
        self.answers_to_user_id_pos_data = {}
        pos_tags_generator = pos_tags_jsons_generator(DATA_DIR)
        for answer_num, ans_pos_tags in pos_tags_generator:
            self.answers_to_user_id_pos_data[answer_num] = ans_pos_tags

        # init model
        self.model = FastText.load_fasttext_format(os.path.join(DATA_DIR, 'ft_pretrained', 'wiki.he.bin'))

        # calculate idf scores for words
        self.idf_scores = IdfScores(self.get_documents(), repair_document)
        self.idf_scores.calculate_idf_scores()

        self.missing_idf = {}
        self.missing_words = {}

    @staticmethod
    def get_documents():
        files_grabbed = []
        data_dir = os.path.join('..', 'data')
        for pattern in ['doctors_articles', '2b_healthy_articles', 'infomed_qa']:
            pattern = os.path.join(data_dir, pattern, '*.txt')
            files_grabbed.extend(glob.glob(pattern))
        return files_grabbed

    def score(self, relevant_contexts, reference_contexts):
        relevant_context_vector = np.zeros((300,))
        reference_context_vector = np.zeros((300,))

        for context in relevant_contexts:
            idf = self.idf_scores.get_idf_score(context)
            if idf:
                context_rep = get_vector_repr_of_word(self.model, context)
                relevant_context_vector += idf * context_rep
            else:
                self.missing_idf[context] = 'relevant word'

        for context in reference_contexts:
            idf = self.idf_scores.get_idf_score(context)
            if idf:
                context_rep = get_vector_repr_of_word(self.model, context)
                reference_context_vector += idf * context_rep
            else:
                self.missing_idf[context] = 'reference word'

        if not relevant_context_vector.any() or not reference_context_vector.any():
            return None

        cos_sim = cosine_similarity([relevant_context_vector], [reference_context_vector])[0][0]
        return cos_sim

    def score_group(self, users_scores, users):
        avgs_nouns = []
        avgs_verbs = []

        for user in users:
            avg_nouns, avg_verbs = self.score_user(user, users_scores)
            avgs_nouns += [avg_nouns]
            avgs_verbs += [avg_verbs]
        avgs_nouns = np.mean([avg for avg in avgs_nouns if not np.isnan(avg)])
        avgs_verbs = np.mean([avg for avg in avgs_verbs if not np.isnan(avg)])
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

    def cos_sim_for_tag(self, word, pos_tag, group):
        if word not in self.reference_tags[pos_tag] or word not in self.relevant_tags[group][pos_tag]:
            if word not in self.reference_tags[pos_tag]:
                self.missing_words[word] = 'reference set'
            return None

        relevant_context = self.relevant_tags[group][pos_tag][word]
        reference_context = self.reference_tags[pos_tag][word]
        score = self.score(relevant_context, reference_context)

        return score

    def cos_sim_score_for_user(self, user, group):
        scores = {}

        for answer_num, users_pos_data in sorted(self.answers_to_user_id_pos_data.items()):
            pos_data = users_pos_data[user]
            ans_noun_scores = []
            ans_verb_scores = []

            if not pos_data['lemmas']:
                continue
            pos_tags = pos_data['posTags']
            words = pos_data['lemmas']

            for word, pos_tag in zip(words, pos_tags):
                if pos_tag == 'noun' or pos_tag == 'verb':
                    score = self.cos_sim_for_tag(word, pos_tag, group)
                    if score and pos_tag == 'noun':
                        ans_noun_scores += [score]
                    elif score and pos_tag == 'verb':
                        ans_verb_scores += [score]

            scores[answer_num] = (ans_noun_scores, ans_verb_scores)
        return scores


def main():
    removed_ids = []
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    df_res = remove_females(df_res, removed_ids)
    df_res = remove_depressed(df_res, removed_ids)
    control = (df_res[df_res['label'] == 'control']['id'].values, 'control')
    patients = (df_res[df_res['label'] == 'patient']['id'].values, 'patients')

    dep_scorer = DependencyCosSimScorer()

    users_scores = {}
    for group, group_name in [control, patients]:
        for user in group:
            user_scores = dep_scorer.cos_sim_score_for_user(user, group_name)
            users_scores[user] = user_scores

    control_scores = dep_scorer.score_group(users_scores, control[0])
    patients_scores = dep_scorer.score_group(users_scores, patients[0])

    print("Control group scores: nouns: {}, verbs: {}".format(*control_scores))
    print("Patients group scores: nouns: {}, verbs: {}".format(*patients_scores))

    print('No idf scores for words {}\n'.format(dep_scorer.missing_idf))
    print('Words missing from sets {}\n'.format(dep_scorer.missing_words))


if __name__ == '__main__':
    main()
