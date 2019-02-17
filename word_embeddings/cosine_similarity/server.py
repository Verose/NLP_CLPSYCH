import json
import os

from flask import Flask, request
from flask_restful import Api
from gensim.models import FastText

from word_embeddings.common.utils import DATA_DIR, read_conf

app = Flask(__name__)
api = Api(app)


@app.route("/word_embeddings", methods=['GET'])
def get():
    data = request.json
    word = data['word']
    is_in = data['isin']

    if is_in:
        return json.dumps(word in model)

    try:
        vector = model[word]
    except KeyError:
        if str.isdecimal(word):
            replacement_word = '<מספר>'
        elif str.isalpha(word):
            replacement_word = '<אנגלית>'
        elif any(i.isdigit() for i in word) and any("\u0590" <= c <= "\u05EA" for c in word):
            replacement_word = '<אות ומספר>'
        else:
            replacement_word = '<לא ידוע>'
        vector = model[replacement_word]
    return json.dumps(vector.tolist())  # todo: think of return embedding anyway


if __name__ == '__main__':
    conf = read_conf()
    print('start loading FastText word embeddings...')
    model = FastText.load_fasttext_format(os.path.join(DATA_DIR, 'ft_pretrained', conf["word_embeddings"]))
    print('finished!')
    app.run(debug=True, use_reloader=False, threaded=True)
