import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_from_tsfile
from aeon.benchmarking import experiments
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

warnings.filterwarnings("ignore")


# Function to update the text widget with classifier output
def update_output(text):
    output_text.config(state=tk.NORMAL)
    output_text.insert(tk.END, text + "\n")
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


# Function to train and test selected classifiers
# Function to train and test selected classifiers
def train_and_test():
    train_filename = train_data_entry.get()
    test_filename = test_data_entry.get()
    selected_classifiers = classifiers_listbox.curselection()
    num_runs = int(num_runs_entry.get())  # Get the number of runs from the entry field

    # Create a table to display accuracy results
    tree.delete(*tree.get_children())  # Clear the table
    avg_accuracy = {}
    avg_balanced_accuracy = {}
    avg_f1_score = {}

    for run in range(num_runs):
        train_data, train_labels = load_from_tsfile(train_filename)
        test_data, test_labels = load_from_tsfile(test_filename)

        # Taking care of data error where 1d data is shown as 3d
        train_data = train_data.reshape(train_data.shape[0], -1)
        test_data = test_data.reshape(test_data.shape[0], -1)

        # Resample the training and testing data
        train_data, train_labels, test_data, test_labels = experiments.stratified_resample(train_data, train_labels,
                                                                                           test_data,
                                                                                           test_labels, run)
        for index in selected_classifiers:
            classifier_name = classifiers[index]
            accuracy = None
            balanced_accuracy = None
            f1 = None

            if classifier_name == "Random Forest":
                random_forest = RandomForestClassifier(n_estimators=100)
                random_forest.fit(train_data, train_labels)
                y_predict = random_forest.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Rocket":
                rocket_classifier = RocketClassifier(num_kernels=2000)
                rocket_classifier.fit(train_data, train_labels)
                y_predict = rocket_classifier.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Hivecotev2":
                hc_2 = HIVECOTEV2(time_limit_in_minutes=0.2)
                hc_2.fit(train_data, train_labels)
                y_predict = hc_2.predict(test_data)
                accuracy = accuracy_score(test_labels, y_predict)
                balanced_accuracy = balanced_accuracy_score(test_labels, y_predict)
                f1 = f1_score(test_labels, y_predict, average='macro')

            elif classifier_name == "Elastic Ensemble":
                knn = KNeighborsTimeSeriesClassifier(distance="msm", n_neighbors=3, weights="distance")
                knn.fit(train_data, train_labels)
                knn_prediction = knn.predict(test_data)
                accuracy = accuracy_score(test_labels, knn_prediction)
                balanced_accuracy = balanced_accuracy_score(test_labels, knn_prediction)
                f1 = f1_score(test_labels, knn_prediction, average='macro')

            elif classifier_name == "Fresh Prince":
                c22cls = Catch22Classifier()
                c22cls.fit(train_data, train_labels)
                fp = FreshPRINCEClassifier()
                fp.fit(train_data, train_labels)
                fp_preds = c22cls.predict(test_data)
                accuracy = accuracy_score(test_labels, fp_preds)
                balanced_accuracy = balanced_accuracy_score(test_labels, fp_preds)
                f1 = f1_score(test_labels, fp_preds, average='macro')

            if accuracy is not None and f1 is not None:
                # Inserting accuracy and F1 score into the table
                avg_accuracy.setdefault(classifier_name, []).append(accuracy)
                avg_balanced_accuracy.setdefault(classifier_name, []).append(balanced_accuracy)
                avg_f1_score.setdefault(classifier_name, []).append(f1)
                tree.insert("", tk.END, values=(f"Run {run + 1}", classifier_name, f"{accuracy:.2f}", f"{f1:.2f}"))

    # Calculate and display average and standard deviation for each classifier
    for classifier_name, accuracies in avg_accuracy.items():
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        avg_bal_acc = np.mean(avg_balanced_accuracy[classifier_name])
        std_bal_acc = np.std(avg_balanced_accuracy[classifier_name])
        avg_f1 = np.mean(avg_f1_score[classifier_name])
        std_f1 = np.std(avg_f1_score[classifier_name])
        update_output(
            f"{classifier_name} Classifier: "
            f"Average Accuracy: {avg_acc:.2f} (Std Dev: {std_acc:.2f}), "
            f"Average Balanced Accuracy: {avg_bal_acc:.2f} (Std Dev: {std_bal_acc:.2f}), "
            f"Average F1 Score: {avg_f1:.2f} (Std Dev: {std_f1:.2f})"
        )

    # Plotting the graph
    plot_graph(avg_accuracy, avg_balanced_accuracy)


def plot_graph(avg_accuracy, avg_balanced_accuracy):
    # Clear the previous graph
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot normal accuracy
    for classifier_name, accuracies in avg_accuracy.items():
        ax1.plot(range(1, len(accuracies) + 1), accuracies, label=classifier_name)
    ax1.set_title("Accuracy of Classifiers Across Runs")
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plot balanced accuracy
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



# Create main Tkinter window

root = tk.Tk()
root.title("Multiple Classifier Selection")
root.geometry("900x1200")

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

# Classifier selection listbox
classifiers = ["Random Forest", "Rocket", "Hivecotev2", "Elastic Ensemble", "Fresh Prince"]
classifiers_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, height=len(classifiers))
for classifier in classifiers:
    classifiers_listbox.insert(tk.END, classifier)
classifiers_listbox.grid(row=2, column=0, columnspan=3)

# Number of runs entry field
num_runs_label = tk.Label(root, text="Number of Runs:")
num_runs_label.grid(row=3, column=0)

num_runs_entry = tk.Entry(root, width=10)
num_runs_entry.grid(row=3, column=1)

# Button to train and test the selected classifiers
submit_button = tk.Button(root, text="Train and Test", command=train_and_test)
submit_button.grid(row=4, column=1)

# Output text area
output_text = ScrolledText(root, height=10, width=100)
output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)
output_text.config(state=tk.DISABLED)

# Create a frame to hold the graph
graph_frame = tk.Frame(root)
graph_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Create a treeview widget to display accuracy and F1 score results
tree = ttk.Treeview(root, columns=("Run", "Classifier", "Accuracy", "F1 Score"), show="headings")
tree.heading("Run", text="Run")
tree.heading("Classifier", text="Classifier")
tree.heading("Accuracy", text="Accuracy")
tree.heading("F1 Score", text="F1 Score")
tree.grid(row=7, column=0, columnspan=4, padx=10, pady=10)

root.mainloop()