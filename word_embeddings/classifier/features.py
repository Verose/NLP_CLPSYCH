import pandas as pd

pos_tags_features = ['noun',
                     'verb',
                     'adjective',
                     'adverb']
cos_sim_features = [
    'cos_sim'
]
sentiment_features = [
    'positive_count',
    'negative_count'
]
word_related_features = [
    'unique_lemma',
    'unique_tokens'
]


def get_features(df_input, column):
    features = pd.DataFrame()
    for col in column:
        regex = 'q_[0-9]+_{}.*'.format(col)
        cols = df_input.filter(regex=regex, axis=1)
        features = pd.concat([features, cols], axis=1)
    return features
