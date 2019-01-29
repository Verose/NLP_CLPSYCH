import os

import numpy as np
import pandas as pd
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

from utils import remove_females, remove_depressed, DATA_DIR
from word_embeddings.cosine_similarity.utils import get_vector_repr_of_word, pos_tags_jsons_generator
from word_embeddings.dependency_parser.generate_relevant_sets import read_relevant_set, read_reference_set

relevant_tags = {
    'control': {
        'noun': read_relevant_set('nouns', 'control'),
        'verb': read_relevant_set('verbs', 'control')
    },
    'patients': {
        'noun': read_relevant_set('nouns', 'patients'),
        'verb': read_relevant_set('verbs', 'patients')
    }
}
reference_tags = {
    'noun': read_reference_set('nouns'),
    'verb': read_reference_set('verbs')
}

model = FastText.load_fasttext_format(os.path.join(DATA_DIR, 'ft_pretrained', 'wiki.he.bin'))


def score(word, contexts):
    scores = []
    word_vector = get_vector_repr_of_word(model, word)

    for context in contexts:
        context_vector = get_vector_repr_of_word(model, context)
        cos_sim = cosine_similarity([word_vector], [context_vector])[0][0]
        scores += [cos_sim]
    return sum(scores) / len(scores)


def score_group(users_scores, users):
    relevant_avg = 0
    reference_avg = 0

    for user in users:
        scores = users_scores[user]
        relevant = np.mean([s['relevant'] for s in scores.values()])
        reference = np.mean([s['reference'] for s in scores.values()])
        relevant_avg += relevant
        reference_avg += reference
    return relevant_avg, reference_avg


def cos_sim_for_tag(word, pos_tag, group):
    if word not in reference_tags[pos_tag] or word not in relevant_tags[group][pos_tag]:
        return None, None

    relevant_context = relevant_tags[group][pos_tag][word]
    reference_context = reference_tags[pos_tag][word]
    relevant_score = score(word, relevant_context)
    reference_score = score(word, reference_context)

    return relevant_score, reference_score


def cos_sim_score_for_user(answers_to_user_id_pos_data, user, group):
    scores = {}

    for answer_num, users_pos_data in sorted(answers_to_user_id_pos_data.items()):
        pos_data = users_pos_data[user]
        cos_sims_relevant = []
        cos_sims_reference = []

        if not pos_data['tokens']:
            continue
        pos_tags = pos_data['posTags']
        words = pos_data['tokens']

        for word, pos_tag in zip(words, pos_tags):
            if pos_tag == 'noun' or pos_tag == 'verb':
                rel_score, ref_score = cos_sim_for_tag(word, pos_tag, group)
                if rel_score and ref_score:
                    cos_sims_relevant += [rel_score]
                    cos_sims_reference += [ref_score]

        scores[answer_num] = {
            'relevant': sum(cos_sims_relevant) / len(cos_sims_relevant) if len(cos_sims_relevant) > 0 else 0,
            'reference': sum(cos_sims_reference) / len(cos_sims_reference) if len(cos_sims_relevant) > 0 else 0
        }
    return scores


def main():
    removed_ids = []
    df_res = pd.read_csv(os.path.join(DATA_DIR, 'features_all.csv'))
    df_res = remove_females(df_res, removed_ids)
    df_res = remove_depressed(df_res, removed_ids)
    control = (df_res[df_res['label'] == 'control']['id'].values, 'control')
    patients = (df_res[df_res['label'] == 'patient']['id'].values, 'patients')

    answers_to_user_id_pos_data = {}
    pos_tags_generator = pos_tags_jsons_generator(DATA_DIR)
    for answer_num, ans_pos_tags in pos_tags_generator:
        answers_to_user_id_pos_data[answer_num] = ans_pos_tags

    users_scores = {}
    for group, name in [control, patients]:
        for user in group:
            user_scores = cos_sim_score_for_user(answers_to_user_id_pos_data, user, name)
            users_scores[user] = user_scores

    control_scores = score_group(users_scores, control[0])
    patients_scores = score_group(users_scores, patients[0])

    print("Control group scores:")
    print("Similarity on relevant set: {}, on reference set: {}".format(*control_scores))
    print("Patients group scores:")
    print("Similarity on relevant set: {}, on reference set: {}".format(*patients_scores))


if __name__ == '__main__':
    main()
