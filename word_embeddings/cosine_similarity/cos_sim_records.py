
class CosSim:
    def __init__(self, userid, question_num, score):
        self.userid = userid
        self.question_num = question_num
        self.score = score

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'id {0}: Q{1}: <{2:.4f}>'.format(self.userid, self.question_num, self.score)


class WindowCosSim:
    def __init__(self, win_size, questions_list):
        self.questions_list = questions_list
        self.window_size = win_size

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        res = "\nwindow {}".format(self.window_size)
        for q in self.questions_list:
            res += '\n{}'.format(str(q))
        return res

