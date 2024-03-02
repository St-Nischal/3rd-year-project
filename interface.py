import tkinter as tk

from tkinter import filedialog


def browse_file():
    filename = filedialog.askopenfilename()
    # Do something with the selected file, like displaying its name
    if filename:
        label.config(text="Selected file: " + filename)
    else:
        label.config(text="No file selected")


# Create the main window
root = tk.Tk()

root.title("3rd year project")
root.geometry("1200x800")

label = tk.Label(root, text='Welcome to my 3rd year project', font=("Arial", 18, "bold"))
label.pack(padx=20, pady=20)

# Create a button to trigger file dialog
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=10)

# Create a label to display selected file
label = tk.Label(root, text="No file selected")
label.pack(pady=5)

# creating a grid layout for buttons
buttonFrame = tk.Frame(root)
buttonFrame.columnconfigure(0, weight=1)
buttonFrame.columnconfigure(1, weight=1)
buttonFrame.columnconfigure(2, weight=1)

btn1 = tk.Button(buttonFrame, text="1", font=("Arial", 18, "bold"))
btn1.grid(row=0, column=0, sticky=tk.W+tk.E)


root.mainloop()
