import tkinter as tk
import pandas as pd
from tkinter import ttk, font
from singleDataset import ClassifierSelectionApp  # Import the other module

class OptionSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Option Selector")
        root.geometry("1000x800")
        
        # Variables to hold user selections
        self.choice_var = tk.StringVar(value="Dataset")
        self.dataset_option_var = tk.StringVar(value="Option 1")
        self.classifier_option_var = tk.StringVar(value="Option A")

        # Frame for the initial choice
        self.choice_frame = ttk.Frame(root, padding="10")
        self.choice_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Define custom font
        self.custom_font = font.Font(family="Helvetica", size=20)

        # Create custom style for the radio buttons
        style = ttk.Style()
        style.configure("Custom.TRadiobutton", font=self.custom_font)

        # Label for the initial choice
        ttk.Label(self.choice_frame, text="Would you like to test your classifier or dataset?", font=("Arial", 25)).grid(row=0, column=0, columnspan=2, sticky='w')

        # Radio buttons for the initial choice
        ttk.Radiobutton(self.choice_frame, text="Dataset", variable=self.choice_var, value="Dataset", command=self.display_options, style="Custom.TRadiobutton").grid(row=1, column=0, sticky='w', pady=10)
        ttk.Radiobutton(self.choice_frame, text="Classifier", variable=self.choice_var, value="Classifier", command=self.display_options, style="Custom.TRadiobutton").grid(row=1, column=1, sticky='w', pady=10)

        # Frame for displaying additional options based on initial choice
        self.options_frame = ttk.Frame(root, padding="10")
        self.options_frame.grid(row=1, column=0, padx=10, pady=5, sticky='nsew')

        # Initial display of options
        self.display_options()

    def display_options(self):
        # Clear existing widgets in the options frame
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        # Check which option is selected and display corresponding options
        if self.choice_var.get() == 'Classifier':
            tk.Label(self.options_frame, text="Select Dataset Option:").grid(row=0, column=0, sticky='w')
            tk.Radiobutton(self.options_frame, text="Option 1", variable=self.dataset_option_var, value="Option 1").grid(row=1, column=0)
        elif self.choice_var.get() == 'Dataset':
            # Instantiate the ClassifierSelectionApp within the options frame
            self.classifier_app = ClassifierSelectionApp(self.options_frame)
        
        self.validate_button = tk.Button(self.root, text="Train Test", command=self.validate_inputs)
        self.validate_button.grid(row=2, column=0, columnspan=2, padx=170, pady=10, sticky="we")

    def validate_inputs(self):
        # Assuming that `train_data_entry` and `test_data_entry` are defined somewhere in the real use case
        train_file = self.train_data_entry.get()
        test_file = self.test_data_entry.get()
        if not train_file:
            self.show_error("Error: Please select a training data file.")
            return False
        if not self.is_time_series(train_file):
            self.show_error("Error: Training data file does not seem to contain valid time series data.")
            return False
        if not test_file:
            self.show_error("Error: Please select a testing data file.")
            return False
        if not self.is_time_series(test_file):
            self.show_error("Error: Testing data file does not seem to contain valid time series data.")
            return False
        custom_classifier_file = self.custom_classifier_entry.get()
        if custom_classifier_file and not custom_classifier_file.endswith(".py"):
            self.show_error("Error: Custom classifier file must end with '.py'.")
            return False
        return True

    def is_time_series(self, file_path):
        try:
            df = pd.read_csv(file_path)
            # Check if any column has datetime-like values
            datetime_col = None
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    datetime_col = col
                    break
                except (ValueError, TypeError):
                    continue
            if datetime_col is None:
                return False
            # Check if the datetime column is sorted
            if not df[datetime_col].is_monotonic_increasing:
                return False
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
        
    def show_error(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.geometry("500x200")
        error_window.title("Error")
        error_label = tk.Label(error_window, text=message, fg="red")
        error_label.pack()
        button = tk.Button(error_window, text="Close", command=error_window.destroy)
        button.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = OptionSelectorApp(root)
    root.mainloop()
