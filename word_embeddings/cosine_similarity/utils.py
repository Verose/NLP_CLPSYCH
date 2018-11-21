
def get_vector_repr_of_word(model, word, logger):
    try:
        return model[word]
    except KeyError:
        if str.isdecimal(word):
            replacement_word = '<מספר>'
        elif str.isalpha(word):
            replacement_word = '<אנגלית>'
        elif any(i.isdigit() for i in word) and any("\u0590" <= c <= "\u05EA" for c in word):
            replacement_word = '<אות ומספר>'
        else:
            replacement_word = '<לא ידוע>'
        logger.debug('word: {} replaced with: {}'.format(word, replacement_word))
        return model[replacement_word]
