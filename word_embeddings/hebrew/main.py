import warnings

import numpy as np
from models import FastTextTrainer, Word2VecTrainer

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.matutils import unitvec


def test(input_model, positive, negative, test_words):
    mean = []
    scores = {}

    for pos_word in positive:
        mean.append(1.0 * np.array(input_model[pos_word]))

    for neg_word in negative:
        mean.append(-1.0 * np.array(input_model[neg_word]))

    # compute the weighted average of all words
    mean = unitvec(np.array(mean).mean(axis=0))

    for word in test_words:
        if word not in positive + negative:
            test_word = unitvec(np.array(input_model[word]))
            # Cosine Similarity
            scores[word] = np.dot(test_word, mean)

    result = sorted(scores, key=scores.get, reverse=True)[:1]
    print(result)
    return result


if __name__ == '__main__':
    TRAIN = False
    word2vec = Word2VecTrainer()
    fasttxt = FastTextTrainer()

    if TRAIN:
        print("Training Word2vec")
        word2vec.train()

        print("Training Fasttext")
        fasttxt.train()

    positive_words = ["מלכה", "גבר"]
    negative_words = ["מלך"]

    # Test Word2vec
    print("Testing Word2vec")
    model = word2vec.get_model()
    assert test(model, positive_words, negative_words, model.wv.vocab) == ["אישה"]

    # Test Fasttext
    print("Testing Fasttext")
    model = fasttxt.get_model()
    assert test(model, positive_words, negative_words, model.words) == ["אישה"]
