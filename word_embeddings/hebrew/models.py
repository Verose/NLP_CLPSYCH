import multiprocessing
import time

import fasttext
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


class FastTextTrainer:
    @staticmethod
    def train(wiki_corpus="wiki_hebrew_corpus.txt", out_model="wiki_hebrew_corpus_fasttext.model", alg="CBOW"):
        start = time.time()

        if alg == "skipgram":
            # Skipgram model
            model = fasttext.skipgram(wiki_corpus, out_model)
        else:
            # CBOW model
            model = fasttext.cbow(wiki_corpus, out_model)
        print(model.words)  # list of words in dictionary
        print(time.time() - start)

    @staticmethod
    def get_model(model="wiki_hebrew_corpus_fasttext.model.bin"):
        return fasttext.load_model(model)


class Word2VecTrainer:
    @staticmethod
    def train(wiki_corpus="wiki_hebrew_corpus.txt", out_model="wiki_hebrew_corpus_word2vec.model"):
        start = time.time()

        model = Word2Vec(LineSentence(wiki_corpus), sg=1,  # 0=CBOW , 1= SkipGram
                         size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())

        # trim unneeded model memory = use (much) less RAM
        model.wv.init_sims(replace=True)
        print(time.time() - start)
        model.save(out_model)

    @staticmethod
    def get_model(model="wiki_hebrew_corpus_word2vec.model"):
        return Word2Vec.load(model)
