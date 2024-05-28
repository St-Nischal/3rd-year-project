from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier

class custom_classifier:
    def __init__(self):
        self.c22cls = Catch22Classifier()
        self.fp = FreshPRINCEClassifier()

    def fit(self, train_data, train_labels):
        self.c22cls.fit(train_data, train_labels)
        self.fp.fit(train_data, train_labels)

    def predict(self, test_data):
        c22_preds = self.c22cls.predict(test_data)
        fp_preds = self.fp.predict(test_data)
        # Combine predictions from both classifiers (you might need to define a combining strategy)
        # For example, you could take the majority vote or the average of the predictions
        combined_preds = (c22_preds + fp_preds) / 2
        return combined_preds