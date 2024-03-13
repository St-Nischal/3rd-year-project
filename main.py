import warnings

from aeon.datasets import load_arrow_head, load_basic_motions

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import RocketClassifier

from aeon.classification.hybrid import HIVECOTEV2

warnings.filterwarnings("ignore")

'''
# loadng the TRAINING data (ie why split="train")
arrow, arrow_labels = load_arrow_head(split="train")
# motions, motions_labels = load_basic_motions(split="train")


# declaring the classifier but not doing anything with it yet
rand_forest = RandomForestClassifier(n_estimators=100)
arrow2d = arrow.squeeze()  # to work out
# important - this is where the classifier is trained (so most of the work happens here)
rand_forest.fit(arrow2d, arrow_labels)

# load the test data
arrow_test, arrow_test_labels = load_arrow_head(split="test", return_type="numpy2d")

# make predictions for each test case
yPredict = rand_forest.predict(arrow_test)

# now that you've got predictions, work out how accurate they were by comparing to the real class labels
print("Accuracy from Random Forest Classifier:" + str(accuracy_score(arrow_test_labels, yPredict)))

# below is repeating the experiment but using a different classifier
rocket = RocketClassifier(num_kernels=2000)
rocket.fit(arrow, arrow_labels)
yPredict1 = rocket.predict(arrow_test)

print("Accuracy from Rocket Classifier:" + str(accuracy_score(arrow_test_labels, yPredict1)))

hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(arrow, arrow_labels)
yPredict2 = hc2.predict(arrow_test)

print("Accuracy from HIVECOTEV2:" + str(accuracy_score(arrow_test_labels, yPredict2)))

'''


def classifiers(filename, classification_type):
    train_data, train_label = filename(split="train", return_type="numpy2d")
    test_data, test_label = filename(split="test", return_type="numpy2d")
    if classification_type == "Random Forest":
        print("hi")


def Ramdom_classifier(train_data, train_label, test_data, test_label):
    random_forest = RandomForestClassifier(
        n_estimators=100)  # declaring the classifier but not doing anything with it yet
    random_forest.fit(train_data, train_label)  # Training classifier
    y_predict = random_forest.predict(test_data)  # make predictions for each test case
    print("Accuracy from Random Forest Classifier:" + str(
        accuracy_score(test_label, y_predict)))  # Working out the accuracy of the Ramdom forest classifier


def Rocket_classifier(train_data, train_label, test_data, test_label):
    rocket_classifier = RocketClassifier(num_kernels=2000)
    rocket_classifier.fit(train_data, train_label)
    y_predict1 = rocket_classifier.predict(test_data)

    print("Accuracy from Rocket classifier Classifier:" + str(accuracy_score(test_label, y_predict1)))


def Hivecotev2_classifier(train_data, train_label, test_data, test_label):
    hc_2 = HIVECOTEV2(time_limit_in_minutes=0.2)
    hc_2.fit(train_data, train_label)
    y_Predict2 = hc_2.predict(test_data)

    print("Accuracy from HIVECOTEV2:" + str(accuracy_score(test_label, y_Predict2)))
