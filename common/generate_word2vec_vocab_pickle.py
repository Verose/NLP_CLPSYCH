import os
import pickle

from gensim.models import KeyedVectors

data_path = os.path.join('data', 'ft_pretrained', "cc.he.300.vec")
model = KeyedVectors.load_word2vec_format(data_path, limit=100000)
vocab = list(model.wv.vocab)
word_to_vec_dict = {word: model[word] for word in vocab}
save_path = os.path.join('data', 'ft_pretrained', 'word2vec2_he_vocab.pickle')

with open(save_path, 'wb') as f:
    pickle.dump(word_to_vec_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
