import warnings

from aeon.datasets import load_arrow_head, load_basic_motions

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import RocketClassifier

from aeon.classification.hybrid import HIVECOTEV2

warnings.filterwarnings("ignore")

# loadng the TRAINING data (ie why split="train")
arrow, arrow_labels = load_arrow_head(split="train")
# motions, motions_labels = load_basic_motions(split="train")

# declaring the classifier but not doing anything with it yet
rand_forest = RandomForestClassifier(n_estimators=100)
arrow2d = arrow.squeeze() # to work out


## important - this is where the classifier is trained (so most of the work happens here)
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
