
class TTest:
    def __init__(self, question_num, tstat, pval):
        self.question_num = question_num
        self.tstat = tstat
        self.pval = pval

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Q{0}: <{1:.4f}, {2:.4f}>'.format(self.question_num, self.tstat, self.pval)


class WindowTTest:
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

