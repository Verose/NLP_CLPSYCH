import datetime
import optparse
import os
import pickle

from gensim.models.wrappers import FastText

from common.utils import DATA_DIR

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--embeddings_file', action="store")
    parser.add_option('--dataset', choices=['rsdd', 'smhd'], default='rsdd', action="store")
    options, remainder = parser.parse_args()

    start = datetime.datetime.now()
    print('Start loading FastText word embeddings at {}'.format(start))
    model = FastText.load_fasttext_format(options.embeddings_file)
    end = datetime.datetime.now()
    print('Finished! took: {}'.format(end - start))

    vocab = list(model.wv.vocab)
    word_to_vec_dict = {word: model[word] for word in vocab}
    del model

    save_path = os.path.join('..', DATA_DIR, 'ft_pretrained', '{}_word2vec.pickle'.format(options.dataset))
    with open(save_path, 'wb') as f:
        pickle.dump(word_to_vec_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
