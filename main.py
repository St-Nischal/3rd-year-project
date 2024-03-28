import os
import warnings

from aeon.datasets import load_arrow_head, load_basic_motions
from anaconda_navigator.static.content import DATA_PATH

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_from_tsfile

import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

warnings.filterwarnings("ignore")


# Function to update the text widget with classifier output
def update_output(text):
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, text + "\n")
    output_text.config(state=tk.DISABLED)


def Random_Forest(train_data, train_label, test_data, test_label):
    random_forest = RandomForestClassifier(
        n_estimators=100)  # declaring the classifier but not doing anything with it yet
    random_forest.fit(train_data, train_label)  # Training classifier
    y_predict = random_forest.predict(test_data)  # make predictions for each test case
    accuracy = accuracy_score(test_label, y_predict)
    update_output("Accuracy from Random Forest Classifier: {:.2f}".format(accuracy))


def Rocket_classifier(train_data, train_label, test_data, test_label):
    rocket_classifier = RocketClassifier(num_kernels=2000)
    rocket_classifier.fit(train_data, train_label)
    y_predict1 = rocket_classifier.predict(test_data)
    accuracy = accuracy_score(test_label, y_predict1)
    update_output("Accuracy from Rocket Classifier: {:.2f}".format(accuracy))


def Hivecotev2_classifier(train_data, train_label, test_data, test_label):
    hc_2 = HIVECOTEV2(time_limit_in_minutes=0.2)
    hc_2.fit(train_data, train_label)
    y_Predict2 = hc_2.predict(test_data)
    accuracy = accuracy_score(test_label, y_Predict2)
    update_output("Accuracy from Hivecotev2 Classifier: {:.2f}".format(accuracy))


def Elastic_Ensemble_classifier(train_data, train_label, test_data, test_label):
    knn = KNeighborsTimeSeriesClassifier(distance="msm", n_neighbors=3, weights="distance")
    knn.fit(train_data, train_label)
    knn_prediction = knn.predict(test_data)
    accuracy = accuracy_score(test_label, knn_prediction)
    update_output("Accuracy from Elastic Ensemble Classifier: {:.2f}".format(accuracy))


def Fresh_Prince_Classifier(train_data, train_label, test_data, test_label):
    c22cls = Catch22Classifier()
    c22cls.fit(train_data, train_label)
    fp = FreshPRINCEClassifier()
    fp.fit(train_data, train_label)
    fp_preds = c22cls.predict(test_data)
    accuracy = accuracy_score(test_label, fp_preds)
    update_output("Accuracy from Fresh Prince Classifier: {:.2f}".format(accuracy))


# Function to handle file submission for training data
def submit_train_data():
    train_filename = filedialog.askopenfilename(title="Select Training Data File")
    train_data_entry.delete(0, tk.END)
    train_data_entry.insert(0, train_filename)


# Function to handle file submission for testing data
def submit_test_data():
    test_filename = filedialog.askopenfilename(title="Select Testing Data File")
    test_data_entry.delete(0, tk.END)
    test_data_entry.insert(0, test_filename)


def train_and_test():
    train_filename = train_data_entry.get()
    test_filename = test_data_entry.get()
    classifier_name = classifier_var.get()

    train_data, train_labels = load_from_tsfile(train_filename)
    test_data, test_labels = load_from_tsfile(test_filename)

    # Taking care of data error where 1d data is shown as 3d
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    if classifier_name == "Random Forest":
        Random_Forest(train_data, train_labels, test_data, test_labels)
    elif classifier_name == "Rocket":
        Rocket_classifier(train_data, train_labels, test_data, test_labels)
    elif classifier_name == "Hivecotev2":
        Hivecotev2_classifier(train_data, train_labels, test_data, test_labels)
    elif classifier_name == "Elastic Ensemble":
        Elastic_Ensemble_classifier(train_data, train_labels, test_data, test_labels)
    elif classifier_name == "Fresh Prince":
        Fresh_Prince_Classifier(train_data, train_labels, test_data, test_labels)


# Create main Tkinter window
root = tk.Tk()
root.title("3rd Year project")
root.geometry("1200x800")

# Training data file selection
train_data_label = tk.Label(root, text="Select Training Data:")
train_data_label.grid(row=0, column=0)

train_data_entry = tk.Entry(root, width=50)
train_data_entry.grid(row=0, column=1)

train_data_button = tk.Button(root, text="Browse", command=submit_train_data)
train_data_button.grid(row=0, column=2)

# Testing data file selection
test_data_label = tk.Label(root, text="Select Testing Data:")
test_data_label.grid(row=1, column=0)

test_data_entry = tk.Entry(root, width=50)
test_data_entry.grid(row=1, column=1)

test_data_button = tk.Button(root, text="Browse", command=submit_test_data)
test_data_button.grid(row=1, column=2)

# Classifier selection dropdown
classifiers = ["Random Forest", "Rocket", "Hivecotev2", "Elastic Ensemble", "Fresh Prince"]
classifier_var = tk.StringVar(root)
classifier_var.set(classifiers[0])  # Default classifier
classifier_label = tk.Label(root, text="Select Classifier:")
classifier_label.grid(row=2, column=0)
classifier_dropdown = tk.OptionMenu(root, classifier_var, *classifiers)
classifier_dropdown.grid(row=2, column=1)

# Button to train and test the classifier
submit_button = tk.Button(root, text="Train and Test", command=train_and_test)
submit_button.grid(row=3, column=1)

output_text = ScrolledText(root, height=10, width=100)
output_text.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
output_text.config(state=tk.DISABLED)

root.mainloop()
