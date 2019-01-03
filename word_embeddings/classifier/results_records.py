
class ResultsRecord:
    def __init__(self, classifier_name,
                 train_accuracy, train_precision, train_recall, train_fscore,
                 test_accuracy, test_precision, test_recall, test_fscore,
                 ):
        self.classifier_name = classifier_name
        self.train_accuracy = train_accuracy
        self.train_precision = train_precision
        self.train_recall = train_recall
        self.train_fscore = train_fscore
        self.test_accuracy = test_accuracy
        self.test_precision = test_precision
        self.test_recall = test_recall
        self.test_fscore = test_fscore


class ClassifierResults:
    def __init__(self, tested_features, num_features, results_list):
        self.tested_features = tested_features
        self.num_features = num_features
        self.results_list = results_list
