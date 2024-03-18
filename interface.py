import warnings

from aeon.datasets import load_arrow_head, load_basic_motions

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_from_arff_file

import tkinter as tk
from tkinter import filedialog
import pandas as pd

warnings.filterwarnings("ignore")


# def browse_file():
#     filename = filedialog.askopenfilename()
#     if filename:
#         try:
#             if filename.endswith('.arff'):
#                 # Load ARFF file and convert to DataFrame
#                 data, meta = arff.loadarff(filename)
#                 df = pd.DataFrame(data)
#
#                 # Optionally, decode string columns from bytes to strings
#                 df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
#
#                 # Prepare features (X) and target variable (y)
#                 X = df.drop(columns=['target_column_name'])  # Replace 'target_column_name' with your target column name
#                 y = df['target_column_name']
#
#                 # Split data into training and testing sets
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#                 # Call Random Forest function
#                 Random_Forest(X_train, y_train, X_test, y_test)
#             else:
#                 df = pd.read_csv(filename)
#                 data = df.iloc[:, 0].tolist()  # Assuming first column is data
#                 labels = df.iloc[:, 1].tolist()
#                 if clicked.get() == 'Random Forest':
#                     Random_Forest(data, labels, data, labels)
#         except Exception as e:
#             print("Error: ", e)


# Create the main window
root = tk.Tk()

root.title("3rd year project")
root.geometry("1200x800")

label = tk.Label(root, text='Welcome to my 3rd year project', font=("Arial", 18, "bold"))
label.pack(padx=20, pady=20)

# # Create a button to trigger file uploading
# browse_button = tk.Button(root, text="Browse", command=browse_file)
# browse_button.pack(pady=10)

# Dropdown menu options
options = [
    "Random Forest",
    "Rocket Classifier",
    "HIVECOTEV2"
]

# datatype of menu text
clicked = tk.StringVar()

# initial menu text
clicked.set("Random Forest")

# Create Dropdown menu
drop = tk.OptionMenu(root, clicked, *options)
drop.pack()


"""
def classifiers(filename, classification_type):
    train_data, train_label = filename(split="train", return_type="numpy2d")
    test_data, test_label = filename(split="test", return_type="numpy2d")
    if classification_type == "Random Forest":
        Ramdom_classifier(train_data, train_label, test_data, test_label)
    elif classification_type == "Ramdom Classifier":
        Ramdom_classifier(train_data, train_label, test_data, test_label)
    elif classification_type == "HIVECOTEV2":
        HIVECOTEV2(train_data, train_label, test_data, test_label)

"""


def Random_Forest(train_data, train_label, test_data, test_label):
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


def Elastic_Ensemble_classifier(train_data, train_label, test_data, test_label):
    knn = KNeighborsTimeSeriesClassifier(distance="msm", n_neighbors=3, weights="distance")
    knn.fit(train_data, train_label)
    knn_prediction = knn.predict(test_data)
    print("Accuracy from Elastic Ensemble:" + str(
        accuracy_score(test_label, knn_prediction)))  # Working out the accuracy of the Elastic Ensemble classifier
    # metrics.accuracy_score(test_label, knn_preds)


def Fresh_Prince_Classifier(train_data, train_label, test_data, test_label):
    fp = FreshPRINCEClassifier()
    c22cls = Catch22Classifier()
    fp.fit(train_data, train_label)
    fp_preds = c22cls.predict(test_data)
    print("Accuracy from Elastic Ensemble:" + str(
        accuracy_score(test_label, fp_preds)))  # Working out the accuracy of the Fresh prince classifier
    # metrics.accuracy_score(y_test, fp_preds)

