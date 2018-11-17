from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity:
    def __init__(self, model, data, window_size):
        self._model = model
        self._data = data
        self._window_size = window_size
        self._labels_to_scores = {}
        self._control_scores = []
        self._patient_scores = []

    def calculate_all_avg_scores(self):
        """
        Calculate the cosine similarity of the entire corpus
        :return:
        """

        # iterate users
        for index, row in self._data.iterrows():
            user_id = row[0]
            label = row[1]
            scores = []

            for answer in row[2:]:
                score = self.avg_answer_score(answer)
                scores += [score]

            avg_user_score = sum(scores) / len(scores)
            self._labels_to_scores[user_id] = (avg_user_score, label)

            if user_id == 'control':
                self._control_scores += [(avg_user_score, user_id)]
            else:
                self._patient_scores += [(avg_user_score, user_id)]

        return self._labels_to_scores

    def calculate_avg_score_for_group(self, group='control'):
        if group == 'control':
            control_scores = [score[0] for score in self._control_scores]
            return sum(control_scores) / len(control_scores)
        else:
            patient_scores = [score[0] for score in self._patient_scores]
            return sum(patient_scores) / len(patient_scores)

    def _get_vector_repr_of_word(self, word):
        if str.isdecimal(word):
            return self._model['<מספר>']
        return self._model[word]

    def avg_answer_score(self, answer):
        """
        Calculate the cosine similarity of an answer
        :param answer:
        :return: cosine similarity score
        """

        scores = []
        answer = answer.split()

        for pos, word in enumerate(answer):
            if pos + self._window_size >= len(answer):
                break

            word_vector = self._get_vector_repr_of_word(word)
            score = 0

            # calculate cosine similarity for window
            for dist in range(1, self._window_size + 1):
                context = answer[pos + dist]
                context_vector = self._get_vector_repr_of_word(context)
                score += cosine_similarity([word_vector], [context_vector])[0][0]

            score /= self._window_size
            scores += [score]

        return sum(scores) / len(scores)
