import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import importlib.util
import tensorflow as tf

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.classification.deep_learning import CNNClassifier
from aeon.classification.dictionary_based import ContractableBOSS
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_from_tsfile
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")


class CustomCNNClassifier(CNNClassifier):
    def _fit(self, X, y):
        # Override the _fit method to set the correct file path for ModelCheckpoint
        self.checkpoint_filepath = "checkpoint.weights.h5"  # Use the correct extension
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        self.callbacks = [checkpoint_callback]
        super()._fit(X, y)


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

def validate_inputs():
    if not train_data_entry.get():
        update_output("Error: Please select a training data file.")
        return False
    if not test_data_entry.get():
        update_output("Error: Please select a testing data file.")
        return False
    if not num_runs_entry.get().isdigit():
        update_output("Error: Number of runs must be a positive integer.")
        return False
    return True

# Function to train and test selected classifiers
def train_and_test():
    if not validate_inputs():
        return

    train_filename = train_data_entry.get()
    test_filename = test_data_entry.get()
    selected_classifiers = classifiers_listbox.curselection()
    num_runs = int(num_runs_entry.get())

    if custom_classifier_entry.get():
        custom_classifier_path = custom_classifier_entry.get()
        custom_classifier_module = load_module_from_file("custom_classifier", custom_classifier_path)
        CustomClassifier = getattr(custom_classifier_module, 'CustomClassifier', None)
        if CustomClassifier:
            classifiers.append(CustomClassifier)

    try:
        train_data, train_labels = load_from_tsfile(train_filename)
        test_data, test_labels = load_from_tsfile(test_filename)
    except Exception as e:
        update_output(f"Error loading data: {e}")
        return

    metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    for run in range(num_runs):
        update_output(f"Run {run + 1}/{num_runs}")

        for idx in selected_classifiers:
            classifier_name = classifiers[idx]
            update_output(f"Training classifier: {classifier_name}")

            classifier = globals()[classifier_name]()
            classifier.fit(train_data, train_labels)
            y_pred = classifier.predict(test_data)

            accuracy = accuracy_score(test_labels, y_pred)
            balanced_acc = balanced_accuracy_score(test_labels, y_pred)
            f1 = f1_score(test_labels, y_pred, average='macro')
            precision = precision_score(test_labels, y_pred, average='macro')
            recall = recall_score(test_labels, y_pred, average='macro')

            metrics['accuracy'].append(accuracy)
            metrics['balanced_accuracy'].append(balanced_acc)
            metrics['f1'].append(f1)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)

            tree.insert("", tk.END, values=(run + 1, classifier_name, accuracy, balanced_acc, f1, precision, recall))

            update_output(f"Results for {classifier_name}: Accuracy={accuracy}, Balanced Accuracy={balanced_acc}, F1 Score={f1}, Precision={precision}, Recall={recall}")

            if classifier_name == "CNNClassifier":
                model = classifier.model
                fig, ax = plt.subplots()
                ax.plot(model.history.history['accuracy'], label='Accuracy')
                ax.plot(model.history.history['val_accuracy'], label='Val Accuracy')
                ax.legend(loc='lower right')
                ax.set_title(f'Accuracy - {classifier_name}')
                canvas = FigureCanvasTkAgg(fig, master=graph_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

    update_output("Training and testing completed.")
    plot_metrics(metrics, selected_classifiers)

def plot_metrics(metrics, selected_classifiers):
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    metric_names = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
    axs = axs.flatten()

    for i, metric in enumerate(metric_names):
        for idx in selected_classifiers:
            classifier_name = classifiers[idx]
            axs[i].plot(range(len(metrics[metric])), metrics[metric], label=classifier_name)
        axs[i].set_title(metric.capitalize())
        axs[i].legend(loc='lower right')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

# Create main application window
root = tk.Tk()
root.title("Time Series Classification Interface")

# Create scrollable frame for all widgets
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)
canvas = tk.Canvas(main_frame)
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create frames for different sections
data_selection_frame = tk.Frame(scrollable_frame)
data_selection_frame.grid(row=0, column=0, sticky="w")

classifier_selection_frame = tk.Frame(scrollable_frame)
classifier_selection_frame.grid(row=1, column=0, sticky="w")

run_control_frame = tk.Frame(scrollable_frame)
run_control_frame.grid(row=2, column=0, sticky="w")

result_display_frame = tk.Frame(scrollable_frame)
result_display_frame.grid(row=3, column=0, sticky="w")

# Data selection widgets
train_data_label = tk.Label(data_selection_frame, text="Select Training Data:")
train_data_label.grid(row=0, column=0)
train_data_entry = tk.Entry(data_selection_frame, width=50)
train_data_entry.grid(row=0, column=1)
train_data_button = tk.Button(data_selection_frame, text="Browse", command=submit_train_data)
train_data_button.grid(row=0, column=2)

test_data_label = tk.Label(data_selection_frame, text="Select Testing Data:")
test_data_label.grid(row=1, column=0)
test_data_entry = tk.Entry(data_selection_frame, width=50)
test_data_entry.grid(row=1, column=1)
test_data_button = tk.Button(data_selection_frame, text="Browse", command=submit_test_data)
test_data_button.grid(row=1, column=2)

custom_classifier_label = tk.Label(data_selection_frame, text="Select Custom Classifier File:")
custom_classifier_label.grid(row=2, column=0)
custom_classifier_entry = tk.Entry(data_selection_frame, width=50)
custom_classifier_entry.grid(row=2, column=1)
custom_classifier_button = tk.Button(data_selection_frame, text="Browse", command=submit_custom_classifier)
custom_classifier_button.grid(row=2, column=2)

# Classifier selection widgets
classifiers = ["RocketClassifier", "HIVECOTEV2", "KNeighborsTimeSeriesClassifier",
               "Catch22Classifier", "FreshPRINCEClassifier", "CNNClassifier",
               "ContractableBOSS", "TimeSeriesForestClassifier"]

classifiers_label = tk.Label(classifier_selection_frame, text="Select Classifiers:")
classifiers_label.grid(row=0, column=0)
classifiers_listbox = tk.Listbox(classifier_selection_frame, selectmode=tk.MULTIPLE, height=len(classifiers))
for classifier in classifiers:
    classifiers_listbox.insert(tk.END, classifier)
classifiers_listbox.grid(row=1, column=0, columnspan=3)

# Run control widgets
num_runs_label = tk.Label(run_control_frame, text="Number of Runs:")
num_runs_label.grid(row=0, column=0)
num_runs_entry = tk.Entry(run_control_frame, width=10)
num_runs_entry.grid(row=0, column=1)
submit_button = tk.Button(run_control_frame, text="Train and Test", command=train_and_test)
submit_button.grid(row=0, column=2)

progress = ttk.Progressbar(run_control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress.grid(row=1, column=0, columnspan=3)

# Result display widgets
output_text = ScrolledText(result_display_frame, height=10, width=100)
output_text.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
output_text.config(state=tk.DISABLED)

graph_frame = tk.Frame(result_display_frame)
graph_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

tree = ttk.Treeview(result_display_frame, columns=("Run", "Classifier", "Accuracy", "F1 Score"), show="headings")
tree.heading("Run", text="Run")
tree.heading("Classifier", text="Classifier")
tree.heading("Accuracy", text="Accuracy")
tree.heading("F1 Score", text="F1 Score")
tree.grid(row=2, column=0, columnspan=4, padx=10, pady=10)

# Start the Tkinter event loop
root.mainloop()