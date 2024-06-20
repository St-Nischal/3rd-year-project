import tkinter as tk
from tkinter import ttk
from tkinter import font


class TabsDisplay:

    def __init__(self,root) -> None:
        self.root = root


        # Create a custom style
        style = ttk.Style()
        tab_font = ('Helvetica', 16, 'bold')
        style.configure('TNotebook.Tab', font=tab_font, padding=[10, 5])

        # Create the Notebook, almost impossible to style notebooks without making custom changes like this
        self.notebook = ttk.Notebook(self.root, style='TNotebook')
        self.notebook.pack(expand=True, fill='both')
        
        # Create and add the tabs
        self.create_tabs()
        self.addFrameNotebook()
        self.addWidgetAverage(self.average)
        self.addWidgetMeanAbsoluteError(self.meanAbsoluteError)
        self.addWidgetMeanSquaredError(self.meanSquaredError)
        self.addWidgetRootMeanSquaredError(self.rootMeanSquaredError)
        self.addWidgetMeanAbsolutePercentageError(self.meanAbsolutePercentageError)
        self.addWidgetSymmetricMeanAbsolutePercentageError(self.symmetricMeanAbsolutePercentageError)
        self.addWidgetMeanAbsoluteScaledError(self.meanAbsoluteScaledError)
        self.addWidgetDynamicTimeWarping(self.dynamicTimeWarping)

    def create_tabs(self):
        # Cannot change style(no padding, or changing fonts, nothing), must do it above where I am creating notebook
        self.average = ttk.Frame(self.notebook)
        self.meanAbsoluteError = ttk.Frame(self.notebook)
        self.meanSquaredError = ttk.Frame(self.notebook)
        self.rootMeanSquaredError = ttk.Frame(self.notebook)
        self.meanAbsolutePercentageError = ttk.Frame(self.notebook)
        self.symmetricMeanAbsolutePercentageError = ttk.Frame(self.notebook)
        self.meanAbsoluteScaledError  = ttk.Frame(self.notebook)
        self.dynamicTimeWarping = ttk.Frame(self.notebook)

    def addFrameNotebook(self):
        # Add the frames to the Notebook as tabs with padding
        self.notebook.add(self.average, text='Average')
        self.notebook.add(self.meanAbsoluteError, text='MAE')
        self.notebook.add(self.meanSquaredError, text='MSE')
        self.notebook.add(self.rootMeanSquaredError, text='rMSE')
        self.notebook.add(self.meanAbsolutePercentageError, text='MAPE')
        self.notebook.add(self.symmetricMeanAbsolutePercentageError, text='sMAPE')
        self.notebook.add(self.meanAbsoluteScaledError, text='MASE')
        self.notebook.add(self.dynamicTimeWarping, text='DTW')

    def addWidgetAverage(self, tab):
        label = ttk.Label(tab, text="This is the Average tab")
        label.pack(pady=20, padx=20)

    def addWidgetMeanAbsoluteError(self, tab):
        label = ttk.Label(tab, text="This is the Mean Absolute Error")
        label.pack(pady=20, padx=20)

    def addWidgetMeanSquaredError(self, tab):
        label = ttk.Label(tab, text="This is the Mean Squared Error")
        label.pack(pady=20, padx=20)
    
    def addWidgetRootMeanSquaredError(self, tab):
        label = ttk.Label(tab, text="This is the Root Mean Squared Error")
        label.pack(pady=20, padx=20)
    
    def addWidgetMeanAbsolutePercentageError(self, tab):
        label = ttk.Label(tab, text="This is the Mean Absolute Percentage Error")
        label.pack(pady=20, padx=20)

    def addWidgetSymmetricMeanAbsolutePercentageError(self, tab):
        label = ttk.Label(tab, text="This is the Symmetric Mean Absolute Percentage Error")
        label.pack(pady=20, padx=20)
    
    def addWidgetMeanAbsoluteScaledError(self, tab):
        label = ttk.Label(tab, text="This is the Mean Absolute Scaled Error")
        label.pack(pady=20, padx=20)
    
    def addWidgetDynamicTimeWarping(self, tab):
        label = ttk.Label(tab, text="This is the Dynamic Time Warping.")
        label.pack(pady=20, padx=20)



if __name__ == "__main__":
    # Create the main application window
    root = tk.Tk()
    root.title("Tkinter Multiple Tabs Example")
    root.geometry("1000x800")

    # Create an instance of the TabsDisplay class
    tabs_display = TabsDisplay(root)

    # Run the main event loop
    root.mainloop()
