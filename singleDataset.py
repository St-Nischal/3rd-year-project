import tkinter as tk
from tkinter import ttk, filedialog

class ClassifierSelectionApp:
    def __init__(self, root):
        self.root = root
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=5)
        style.configure("TLabel", font=("Arial", 12))

        # Main frame
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid
        for i in range(8):
            main_frame.columnconfigure(i, weight=1)
        for i in range(6):
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
            label = ttk.Label(main_frame, text=category, font=("Arial", 10, "bold"))
            label.grid(row=2, column=i, padx=5, pady=5)

        # Row Labels and Checkbuttons
        self.check_vars = []
        rows = [
            ["Row e.g", "Row 1-2", "Row 1-3", "Row 1-4", "Row 1-5", "Row 1-6", "Row 1-7", "Row 1-8"],
            ["Row 2-1", "Row 2-2", "Row 2-3", "Row 2-4", "Row 2-5", "Row 2-6", "Row 2-7", "Row 2-8"],
            ["Row 3-1", "Row 3-2", "Row 3-3", "Row 3-4", "Row 3-5", "Row 3-6", "Row 3-7", "Row 3-8"]
        ]

        for r, row in enumerate(rows):
            row_vars = []
            for c, item in enumerate(row):
                var = tk.BooleanVar()
                checkbutton = ttk.Checkbutton(main_frame, text=item, variable=var)
                checkbutton.grid(row=r+3, column=c, padx=5, pady=5)
                row_vars.append(var)
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