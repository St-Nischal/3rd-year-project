import tkinter as tk
from tkinter import ttk, filedialog

class ClassifierSelectionApp:
    def __init__(self, root):
        self.root = root
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 11), padding=5)
        style.configure("TLabel", font=("Arial", 11))

        # Main frame
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

       # Training Data
        self.label_train = ttk.Label(main_frame, text="Select Training Data:", style="TLabel")
        self.label_train.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # Displaying the location of the training data
        self.train_data_entry = tk.Entry(main_frame, width=25)
        self.train_data_entry.grid(row=1, column=0)

        self.button_train = ttk.Button(main_frame, text="Browse", command=self.browse_train, style="TButton")
        self.button_train.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        # Testing Data
        self.label_test = ttk.Label(main_frame, text="Select Testing Data:", style="TLabel")
        self.label_test.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # Displaying the location of the testing data
        self.test_data_entry = tk.Entry(main_frame, width=25)
        self.test_data_entry.grid(row=1, column=1)

        self.button_test = ttk.Button(main_frame, text="Browse", command=self.browse_test, style="TButton")
        self.button_test.grid(row=2, column=1, padx=5, pady=5, sticky='w')

        # Category frame
        category_frame = ttk.Frame(root)
        category_frame.grid(row=1, column=0)

        # better use a dictionary to store the category data
        categories = {
            "Convolution based": ["RocketClassifier", "ArsenalClassifier"],
            "Shapelet based": ["MrSQMClassifier", "RDSTClassifier", "SASTClassifier"],
            "Dictionary based": ["MUSE", "WEASEL", "BOSS Ensemble", "Contractable BOSS", "Temporal Dictionary Ensemble"],
            "Interval based": ["CIFClassifier", "DrCIFClassifier", "RISEClassifier", "STSFClassifier", "TSFClassifier"],
            "Feature based": ["Catch22Classifier", "FreshPRINCEClassifier"],
            "Distance based": ["K-NNClassifier", "Elastic Ensemble"],
            "Deep learning": ["CNNClassifier"],
            "Hybrid": ["HIVECOTEV2Classifier"],
        }
        # calculate number of columns for two rows
        columns = int(len(categories) / 2 + 0.5)
        self.check_vars = {}
        for i, (category, items) in enumerate(categories.items()):
            self.check_vars[category] = []
            # determine the row and column
            row, col = divmod(i, columns)
            # create a frame for this category
            frame = ttk.Frame(category_frame)
            frame.grid(row=row, column=col, sticky='nsew', padx=5, pady=5)
            ttk.Label(frame, text=category, font=('Arial', 11, 'bold')).pack(padx=5, pady=5, anchor='w')
            for item in items:
                var = tk.BooleanVar()
                ttk.Checkbutton(frame, text=item, variable=var).pack(anchor='w', padx=5, pady=5)
                self.check_vars[category].append(var)

    #Function for displaying the location of the data in computer
    def browse_train(self):
        file_path = filedialog.askopenfilename()
        print(f"Training data selected: {file_path}")
        self.train_data_entry.delete(0, tk.END)
        self.train_data_entry.insert(0, file_path)

    def browse_test(self):
        file_path = filedialog.askopenfilename()
        print(f"Testing data selected: {file_path}")
        self.test_data_entry.delete(0, tk.END)
        self.test_data_entry.insert(0, file_path)

    # Function to handle file submission for training data
    def submit_train_data(self):
        train_filename = filedialog.askopenfilename(title="Select Training Data File")
        self.train_data_entry.delete(0, tk.END)
        self.train_data_entry.insert(0, train_filename)

    # Function to handle file submission for testing data
    def submit_test_data(self):
        test_filename = filedialog.askopenfilename(title="Select Testing Data File")
        self.test_data_entry.delete(0, tk.END)
        self.test_data_entry.insert(0, test_filename)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierSelectionApp(root)
    root.mainloop()