from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class IdfScores:
    def __init__(self, file_names, repair_document_cb):
        self._files = file_names
        self._repair_document_cb = repair_document_cb
        self._v = TfidfVectorizer()
        self._idf = None

    def calculate_idf_scores(self):
        corpus = []

        for file in self._files:
            with open(file, encoding="utf-8") as f:
                document = f.read()
                document = self._repair_document_cb(document)
                corpus += [document]

        self._v.fit_transform(corpus)

        # normalize idf vector
        scaler = MinMaxScaler()
        # TODO: should scale so it sums to 1?
        self._idf = scaler.fit_transform(self._v.idf_.reshape(-1, 1)).squeeze()
        # self._idf = self._v.idf_/self._v.idf_.sum(axis=0, keepdims=1)

    def get_idf_score(self, word):
        return self._idf[self._v.vocabulary_[word]] if word in self._v.vocabulary_ else None
