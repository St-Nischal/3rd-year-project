README

A Data Science Pipeline to Automatically Evaluate Time Series Classification Problems.

This application is a Tkinter-based GUI that allows users to select and benchmark multiple classifiers on time series data. Users can load training and testing datasets, choose classifiers, set the number of runs, and view the results of the classifiers' performance.
Features

    Load Training and Testing Data: Select training and testing data files in the .ts format.
    Select Classifiers: Choose from a list of pre-defined classifiers including Random Forest, Rocket, HiveCOTEV2, Elastic Ensemble, Fresh Prince, CNN, CBoss, Time Series Forest, and 	a custom classifier.
    Set Number of Runs: Specify the number of times to run each classifier for better performance evaluation.
    View Results: Display accuracy, balanced accuracy, F1 score, and precision for each classifier in a table.
    Plot Results: Generate plots for accuracy and balanced accuracy across multiple runs.

Requirements

    Python 3.7+
    Required Python libraries:
        tkinter
        matplotlib
        numpy
        tensorflow
        scikit-learn
        aeon

Installation

    Clone this repository. "https://github.com/St-Nischal/3rd-year-project.git"
    Install the required libraries using pip and condo:

    sh

	conda create -n aeon-env -c conda-forge aeon
	conda activate aeon-env
	pip install tinter(Mac only, windows version comes with it pre installed), tenser flow 

Run the script:

sh

    python interface.py

Usage

    Select Training Data:
        Click the "Browse" button next to "Select Training Data" and choose your training data file in .ts format.

    Select Testing Data:
        Click the "Browse" button next to "Select Testing Data" and choose your testing data file in .ts format.

    Select Custom Classifier File (Optional):
        Click the "Browse" button next to "Select Custom Classifier File" and choose your custom classifier file if needed, file in .py format.

    Choose Classifiers:
        Select the classifiers you want to benchmark from the list by holding Ctrl (or Cmd on Mac) and clicking on the classifier names.

    Set Number of Runs:
        Enter the number of times you want each classifier to run in the "Number of Runs" entry field.

    Train and Test:
        Click the "Train and Test" button to start the benchmarking process.

    View Results:
        The output text area will display the average metrics for each classifier.
        The table will show the accuracy and F1 score for each run.
        The graph frame will display plots of accuracy and balanced accuracy across the runs.

Code Structure

    Main Functions:
        validate_inputs(): Validates the user inputs.
        update_output(text): Updates the output text area.
        submit_train_data(): Handles file selection for training data.
        submit_test_data(): Handles file selection for testing data.
        submit_custom_classifier(): Handles file selection for custom classifier.
        train_and_test(): Main function to train and test the selected classifiers and display the results.
        plot_graph(avg_accuracy, avg_balanced_accuracy): Plots accuracy and balanced accuracy of classifiers across runs.

    Custom Classifier:
        CustomCNNClassifier: Extends the CNNClassifier from aeon to include a checkpoint callback.

Notes

    Ensure the datasets are in the correct format (.ts).
    If using a custom classifier, make sure it is properly defined and the setup() function is implemented to return an instance of the classifier.

Example

Here's an example of how to run the application:

   1. Load your training and testing datasets.
   2. Select the classifiers you want to benchmark.
   3. Specify the number of runs.
   4. Click "Train and Test" to start the benchmarking process.
   5. View the results in the output text area, table, and graph.