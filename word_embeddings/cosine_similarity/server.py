import datetime
import json
import logging
import optparse

from flask import Flask, request
from flask_restful import Api
from gensim.models.wrappers import FastText

from word_embeddings.common.utils import load_model, get_words

app = Flask(__name__)
api = Api(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


@app.route("/word_embeddings", methods=['GET'])
def get():
    """
    words: list of words
    isin: whether or not to check if the first word exists in the model
    :return: if isin is set - returns whether the first word exists in the model.
    otherwise returns a list of vector representations for each word. keeps order. assumes all words exist.
    """
    data = request.json
    words = data['words']
    is_in = data['isin']

    if is_in:
        return json.dumps(words[0] in model)

    vectors = []
    for word in words:
        vectors.append(model[word].tolist())
    return json.dumps(vectors)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--embeddings_file', action="store")
    parser.add_option('--is_rsdd', action="store_true", default=False)
    options, remainder = parser.parse_args()

    start = datetime.datetime.now()
    print('Start loading FastText word embeddings at {}'.format(start))
    if options.is_rsdd:
        model = FastText.load_fasttext_format(options.embeddings_file)
    else:
        model = load_model(get_words(), options.embeddings_file)
    end = datetime.datetime.now()
    print('Finished! took: {}'.format(end - start))

    app.run(use_reloader=False, threaded=True)
