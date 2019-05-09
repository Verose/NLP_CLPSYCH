import datetime
import json
import logging
import optparse
import os
import pickle

from flask import Flask, request
from flask_restful import Api

from word_embeddings.common.utils import load_model, get_words, DATA_DIR

app = Flask(__name__)
api = Api(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


@app.route("/word_embeddings/is_in", methods=['GET'])
def get_is_in():
    data = request.json
    words = data['words']
    result_dict = {}

    for word in words:
        result_dict[word] = word in model

    return json.dumps(result_dict)


@app.route("/word_embeddings/vectors", methods=['GET'])
def get_vectors():
    data = request.json
    words = data['words']
    vectors = []

    for word in words:
        try:
            vectors.append(model[word].tolist())
        except SystemError as e:
            log.error(f'An error occurred in <get_vectors>: \n{e}')

    return json.dumps(vectors)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--embeddings_file', action="store")
    parser.add_option('--is_rsdd', action="store_true", default=False)
    options, remainder = parser.parse_args()

    start = datetime.datetime.now()
    print('Start loading FastText word embeddings at {}'.format(start))
    if options.is_rsdd:
        rsdd_data_path = os.path.join('..', DATA_DIR, 'ft_pretrained', 'rsdd_word2vec.pickle')
        model = pickle.load(rsdd_data_path)
    else:
        model = load_model(get_words(), options.embeddings_file)
    end = datetime.datetime.now()
    print('Finished! took: {}'.format(end - start))

    app.run(use_reloader=False, threaded=True)
