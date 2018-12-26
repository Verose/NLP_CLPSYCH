
class PosData:
    def __init__(self, user, q_num, nouns, verbs, adjectives, adverbs, cossim_score=None, sentiment=None):
        self.user = user
        self.q_num = q_num
        self.nouns = nouns
        self.verbs = verbs
        self.adjectives = adjectives
        self.adverbs = adverbs
        self.cossim_score = cossim_score
        self.sentiment = sentiment

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'q_num{}: ' \
               '<user: {}, nouns: {}, verbs: {}, adjectives: {}, adverbs: {}, cossim_score: {}, sentiment: {}>'.format(
                self.user,
                self.q_num,
                self.nouns,
                self.verbs,
                self.adjectives,
                self.adverbs,
                self.cossim_score,
                self.sentiment
                )
