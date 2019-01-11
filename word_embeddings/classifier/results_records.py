class TestResults:
    def __init__(self, tested_features, num_features, results_list):
        self.tested_features = tested_features
        self.num_features = num_features
        self.results_list = results_list


class AnswersResults:
    def __init__(self, answer_number, results_list):
        self.answer_number = answer_number
        self.results_list = results_list


class ResultsRecord:
    def __init__(self, classifier_name, accuracy, precision, recall, f1):
        self.classifier_name = classifier_name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
