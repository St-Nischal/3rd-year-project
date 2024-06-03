import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import importlib.util
import seaborn as sns
import tensorflow as tf

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.transformations.collection.feature_based import Catch22
from aeon.classification.deep_learning import CNNClassifier
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_from_tsfile
from aeon.benchmarking import experiments
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score,  confusion_matrix

warnings.filterwarnings("ignore")

def validate_inputs():
    if not train_data_entry.get():
        update_output("Error: Please select a training data file.")
        return False
    if not train_data_entry.get().endswith(".ts"):
        update_output("Error: Training data file must end with '.ts'.")
        return False
    if not test_data_entry.get():
        update_output("Error: Please select a testing data file.")
        return False
    if not test_data_entry.get().endswith(".ts"):
        update_output("Error: Testing data file must end with '.ts'.")
        return False
    if not num_runs_entry.get().isdigit() or int(num_runs_entry.get()) <= 0:
        update_output("Error: Number of runs must be a positive integer greater than 0.")
        return False
    if custom_classifier_entry.get() and not custom_classifier_entry.get().endswith(".py"):
        update_output("Error: Custom classifier file must end with '.py'.")
        return False
    if len(classifiers_listbox.curselection()) == 0:
        update_output("Error: Please select at least one classifier.")
        return False
    return True


# Function to update the text widget with classifier output
def update_output(text):
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, text + "\n")
    output_text.config(state=tk.DISABLED)

# Function to clear the output text widget
def clear_output():
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.config(state=tk.DISABLED)

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

# Function to dynamically load a module from a file path
def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Function to handle file submission for custom classifier
def submit_custom_classifier():
    custom_classifier_filename = filedialog.askopenfilename(title="Select Custom Classifier File")
    custom_classifier_entry.delete(0, tk.END)
    custom_classifier_entry.insert(0, custom_classifier_filename)

def train_and_test():
    clear_output()
    if not validate_inputs():
        return
    train_filename = train_data_entry.get()
    test_filename = test_data_entry.get()
    selected_classifiers = classifiers_listbox.curselection()
    num_runs = int(num_runs_entry.get())  # Get the number of runs from the entry field
    custom_classifier_filename = custom_classifier_entry.get()

    # Creating a table to display accuracy results
    tree.delete(*tree.get_children())  # Clear the table
    avg_accuracy = {}
    avg_balanced_accuracy = {}
    avg_f1_score = {}
    avg_precision = {}

    for run in range(num_runs):
        train_data, train_labels = load_from_tsfile(train_filename)
        test_data, test_labels = load_from_tsfile(test_filename)

        # Taking care of data error where 1d data is shown as 3d
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

        # Resampling the training and testing data
        train_data, train_labels, test_data, test_labels = experiments.stratified_resample(train_data, train_labels, test_data, test_labels, run)
        for index in selected_classifiers:
            classifier_name = classifiers[index]
            accuracy = None
            balanced_accuracy = None
            f1 = None
            precision = None

            # Calling classifiers from AEON

            if classifier_name == "Random Forest":
                random_forest = RandomForestClassifier(n_estimators=100)
                random_forest.fit(train_data, train_labels)
                y_predict = random_forest.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')
                precision = precision_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Rocket":
                rocket_classifier = RocketClassifier(num_kernels=2000)
                rocket_classifier.fit(train_data, train_labels)
                y_predict = rocket_classifier.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')
                precision = precision_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Hivecotev2":
                hc_2 = HIVECOTEV2(time_limit_in_minutes=0.2)
                hc_2.fit(train_data, train_labels)
                y_predict = hc_2.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')
                precision = precision_score(test_labels, y_predict, average='macro')

            elif classifier_name == "CNN":
                cnn = CNNClassifier()
                cnn.fit(train_data, train_labels)
                y_predict = cnn.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')
                precision = precision_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Elastic Ensemble":
                knn = KNeighborsTimeSeriesClassifier(distance="msm", n_neighbors=3, weights="distance")
                knn.fit(train_data, train_labels)
                knn_prediction = knn.predict(test_data)
                accuracy = accuracy_score(test_labels, knn_prediction)
                balanced_accuracy = balanced_accuracy_score(test_labels, knn_prediction)
                f1 = f1_score(test_labels, knn_prediction, average='macro')
                precision = precision_score(test_labels, knn_prediction, average='macro')

            elif classifier_name == "Fresh Prince":
                c22cls = Catch22Classifier()
                c22cls.fit(train_data, train_labels)
                fp = FreshPRINCEClassifier()
                fp.fit(train_data, train_labels)
                fp_preds = c22cls.predict(test_data)
                accuracy = accuracy_score(test_labels, fp_preds)
                balanced_accuracy = balanced_accuracy_score(test_labels, fp_preds)
                f1 = f1_score(test_labels, fp_preds, average='macro')
                precision = precision_score(test_labels, fp_preds, average='macro')

            elif classifier_name == "CBoss":
                cboss = ContractableBOSS(n_parameter_samples=250, max_ensemble_size=50, random_state=47)
                cboss.fit(train_data, train_labels)
                cboss_preds = cboss.predict(test_data)
                accuracy = accuracy_score(test_labels, cboss_preds)
                balanced_accuracy = balanced_accuracy_score(test_labels, cboss_preds)
                f1 = f1_score(test_labels, cboss_preds, average='macro')
                precision = precision_score(test_labels, cboss_preds, average='macro')

            elif classifier_name == "Time Series Forest":
                tsf = TimeSeriesForestClassifier(n_estimators=50, random_state=47)
                tsf.fit(train_data, train_labels)
                tsf_preds = tsf.predict(test_data)
                accuracy = accuracy_score(test_labels, tsf_preds)
                balanced_accuracy = balanced_accuracy_score(test_labels, tsf_preds)
                f1 = f1_score(test_labels, tsf_preds, average='macro')
                precision = precision_score(test_labels, tsf_preds, average='macro')

            elif classifier_name == "Custom" and custom_classifier_filename:
                custom_module = load_module_from_file("custom_classifier", custom_classifier_filename)
                custom_classifier = custom_module.setup()  # Calling the setup function to get the classifier instance
                custom_classifier.fit(train_data, train_labels)
                custom_preds = custom_classifier.predict(test_data)
                accuracy = accuracy_score(test_labels, custom_preds)
                balanced_accuracy = balanced_accuracy_score(test_labels, custom_preds)
                f1 = f1_score(test_labels, custom_preds, average='macro')
                precision = precision_score(test_labels, custom_preds, average='macro')

            if accuracy is not None and f1 is not None and precision is not None:
                # Inserting accuracy, precision, and F1 score into the table
                avg_accuracy.setdefault(classifier_name, []).append(accuracy)
                avg_balanced_accuracy.setdefault(classifier_name, []).append(balanced_accuracy)
                avg_f1_score.setdefault(classifier_name, []).append(f1)
                avg_precision.setdefault(classifier_name, []).append(precision)
                tree.insert("", tk.END, values=(f"Run {run + 1}", classifier_name, f"{accuracy:.2f}", f"{f1:.2f}", f"{precision:.2f}"))

    # Calculating and displaying average and standard deviation for each classifier
    for classifier_name, accuracies in avg_accuracy.items():
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_bal_acc = np.mean(avg_balanced_accuracy[classifier_name])
        std_bal_acc = np.std(avg_balanced_accuracy[classifier_name])
        avg_f1 = np.mean(avg_f1_score[classifier_name])
        std_f1 = np.std(avg_f1_score[classifier_name])
        avg_prec = np.mean(avg_precision[classifier_name])
        std_prec = np.std(avg_precision[classifier_name])
        update_output(
            f"{classifier_name} Classifier: "
            f"Average Accuracy: {avg_acc:.2f} (Std Dev: {std_acc:.2f}), "
            f"Average Balanced Accuracy: {avg_bal_acc:.2f} (Std Dev: {std_bal_acc:.2f}), "
            f"Average F1 Score: {avg_f1:.2f} (Std Dev: {std_f1:.2f}), "
            f"Average Precision: {avg_prec:.2f} (Std Dev: {std_prec:.2f})\n"
        )

    # Plotting the graph
    plot_graph(avg_accuracy, avg_balanced_accuracy, num_runs)
    
def plot_graph(avg_accuracy, avg_balanced_accuracy, num_runs):
    # Clearing the previous graph, when same window used to run again
    for widget in graph_frame.winfo_children():
        widget.destroy()

    if num_runs == 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Getting a colormap
        cmap = plt.get_cmap("tab10")
        classifier_names = list(avg_accuracy.keys())
        colors = [cmap(i) for i in range(len(classifier_names))]

        # Plotting bar chart for single run accuracy
        ax1 = axes[0]
        accuracies = [avg_accuracy[classifier_name][0] for classifier_name in classifier_names]
        ax1.bar(classifier_names, accuracies, color=colors)
        ax1.set_title("Accuracy of Classifiers for Single Run")
        ax1.set_ylabel("Accuracy")
        ax1.set_xticklabels(classifier_names, rotation=45, ha="right")
        ax1.grid(True)

        # Plotting bar chart for single run balanced accuracy
        ax2 = axes[1]
        bal_accuracies = [avg_balanced_accuracy[classifier_name][0] for classifier_name in classifier_names]
        ax2.bar(classifier_names, bal_accuracies, color=colors)
        ax2.set_title("Balanced Accuracy of Classifiers for Single Run")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.set_xticklabels(classifier_names, rotation=45, ha="right")
        ax2.grid(True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotting normal accuracy
        for classifier_name, accuracies in avg_accuracy.items():
            ax1.plot(range(1, len(accuracies) + 1), accuracies, label=classifier_name)
        ax1.set_title("Accuracy of Classifiers Across Runs")
        ax1.set_xlabel("Run")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Plotting balanced accuracy
        for classifier_name, bal_accuracies in avg_balanced_accuracy.items():
            ax2.plot(range(1, len(bal_accuracies) + 1), bal_accuracies, label=classifier_name)
        ax2.set_title("Balanced Accuracy of Classifiers Across Runs")
        ax2.set_xlabel("Run")
        ax2.set_ylabel("Balanced Accuracy")
        ax2.legend()
        ax2.grid(True)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Createing main Tkinter window
root = tk.Tk()
root.title("Multiple Classifier Selection")
root.geometry("1050x1200")

# Createing a canvas widget and attach a scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
# Adding widgets to the scrollable frame

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

canvas.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

# Training data file selection
train_data_label = tk.Label(scrollable_frame, text="Select Training Data:")
train_data_label.grid(row=0, column=0)

train_data_entry = tk.Entry(scrollable_frame, width=70)
train_data_entry.grid(row=0, column=1)

train_data_button = tk.Button(scrollable_frame, text="Browse", command=submit_train_data)
train_data_button.grid(row=0, column=2)

# Testing data file selection
test_data_label = tk.Label(scrollable_frame, text="Select Testing Data:")
test_data_label.grid(row=1, column=0)

test_data_entry = tk.Entry(scrollable_frame, width=70)
test_data_entry.grid(row=1, column=1)

test_data_button = tk.Button(scrollable_frame, text="Browse", command=submit_test_data)
test_data_button.grid(row=1, column=2)

# Custom classifier file selection
custom_classifier_label = tk.Label(scrollable_frame, text="Select Custom Classifier File:")
custom_classifier_label.grid(row=2, column=0)

custom_classifier_entry = tk.Entry(scrollable_frame, width=70)
custom_classifier_entry.grid(row=2, column=1)

custom_classifier_button = tk.Button(scrollable_frame, text="Browse", command=submit_custom_classifier)
custom_classifier_button.grid(row=2, column=2)

# Classifier selection listbox
classifiers = ["Random Forest", "Rocket", "Hivecotev2", "Elastic Ensemble", "Fresh Prince","CNN","CBoss","Time Series Forest", "Custom"]
classifiers_listbox = tk.Listbox(scrollable_frame, selectmode=tk.MULTIPLE, height=len(classifiers))
for classifier in classifiers:
    classifiers_listbox.insert(tk.END, classifier)
classifiers_listbox.grid(row=3, column=0, columnspan=3)

num_runs_label = tk.Label(scrollable_frame, text="Number of Runs:")
num_runs_label.grid(row=4, column=0)

# Function to validate that input is a whole number
def validate_whole_number(P):
    if P.isdigit() or P == "":
        return True
    return False

vcmd = (scrollable_frame.register(validate_whole_number), '%P')

num_runs_entry = tk.Entry(scrollable_frame, width=10, validate="key", validatecommand=vcmd)
num_runs_entry.grid(row=4, column=1)

# Button to train and test the selected classifiers
submit_button = tk.Button(scrollable_frame, text="Train and Test", command=train_and_test)
submit_button.grid(row=5, column=1)

# Output text area
output_text = ScrolledText(scrollable_frame, height=10, width=120)
output_text.grid(row=6, column=0, columnspan=3, padx=10, pady=10)
output_text.config(state=tk.DISABLED)

# Creating a frame to hold the graph
graph_frame = tk.Frame(scrollable_frame)
graph_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Creating a treeview widget to display accuracy and F1 score results
tree = ttk.Treeview(scrollable_frame, columns=("Run", "Classifier", "Accuracy", "F1 Score"), show="headings")
tree.heading("Run", text="Run")
tree.heading("Classifier", text="Classifier")
tree.heading("Accuracy", text="Accuracy")
tree.heading("F1 Score", text="F1 Score")
tree.grid(row=8, column=0, columnspan=4, padx=10, pady=10)

root.mainloop()
