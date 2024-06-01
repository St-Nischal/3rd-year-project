from sklearn.ensemble import RandomForestClassifier


def setup():
    return custom_classifier()

class custom_classifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)

    def fit(self, train_data, train_labels):
        self.classifier.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.classifier.predict(test_data)
    
    
