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

        # Configure grid
        for i in range(8):  # Number of columns
            main_frame.columnconfigure(i, weight=1)
        for i in range(8):  # Increased number of rows
            main_frame.rowconfigure(i, weight=1)

        # Training Data
        self.label_train = ttk.Label(main_frame, text="Select Training Data:", style="TLabel")
        self.label_train.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.button_train = ttk.Button(main_frame, text="Browse", command=self.browse_train, style="TButton")
        self.button_train.grid(row=1, column=0, padx=5, pady=5)

        # Testing Data
        self.label_test = ttk.Label(main_frame, text="Select Testing Data:", style="TLabel")
        self.label_test.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        self.button_test = ttk.Button(main_frame, text="Browse", command=self.browse_test, style="TButton")
        self.button_test.grid(row=1, column=1, padx=5, pady=5)

        # Category Labels
        categories = ["Convolution based", "Deep learning", "Dictionary based", 
                      "Distance based", "Feature based", "Interval based", 
                      "Shapelet based", "Hybrid"]
        for i, category in enumerate(categories):
            label = ttk.Label(main_frame, text=category, font=("Arial", 11, "bold"))
            label.grid(row=2, column=i, padx=5, pady=5, sticky='w')  # Align to the left

        # Row Labels and Checkbuttons
        self.check_vars = []
        rows_top = [
            ["Rocket Classifier", "CNN ", "MUSE", "K-NN", "Catch22", "Canonical Interval Forest", "Random Forest", "HIVECOTEV2"],
            ["Arsenal", "", "WEASEL", "Elastic Ensemble", "Fresh PRINCE", "DrCIF", "", ""],
            ["", "", "BOSS Ensemble", "", "", "RandomIntervalSpectralEnsemble", "", ""]
        ]
        rows_bottom = [
            ["", "", "Contractable BOSS", "", "", "SupervisedTimeSeriesForest", "", ""],
            ["", "", "Individual BOSS", "", "", "TimeSeriesForestClassifier", "", ""],
            ["", "", "Temporal Dictionary Ensemble", "", "", "", "", ""]
        ]

        # Determine the maximum width for each column
        max_widths = [max(len(row[i]) for row in rows_top + rows_bottom if i < len(row) and row[i]) for i in range(len(categories))]

        # Place the top half of the rows in the grid
        for r, row in enumerate(rows_top):
            row_vars = []
            for c, item in enumerate(row):
                if item:
                    var = tk.BooleanVar()
                    checkbutton = ttk.Checkbutton(main_frame, text=item, variable=var, width=max_widths[c])
                    checkbutton.grid(row=r+3, column=c, padx=5, pady=5, sticky='w')  # Align to the left
                    row_vars.append(var)
                else:
                    row_vars.append(None)
            self.check_vars.append(row_vars)

        # Place the bottom half of the rows in the grid, offset by the number of rows in rows_top
        for r, row in enumerate(rows_bottom):
            row_vars = []
            for c, item in enumerate(row):
                if item:
                    var = tk.BooleanVar()
                    checkbutton = ttk.Checkbutton(main_frame, text=item, variable=var, width=max_widths[c])
                    checkbutton.grid(row=r+3+len(rows_top), column=c, padx=5, pady=5, sticky='w')  # Align to the left
                    row_vars.append(var)
                else:
                    row_vars.append(None)
            self.check_vars.append(row_vars)

    def browse_train(self):
        file_path = filedialog.askopenfilename()
        print(f"Training data selected: {file_path}")
        
    def browse_test(self):
        file_path = filedialog.askopenfilename()
        print(f"Testing data selected: {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierSelectionApp(root)
    root.mainloop()