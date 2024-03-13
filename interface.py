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


def show():
    label.config(text=clicked.get())

# Dropdown menu options
options = [
    "Random Forest",
    "Rocket Classifier",
    "HIVECOTEV2"
]

# datatype of menu text
clicked = tk.StringVar()

# initial menu text
clicked.set("Monday")

# Create Dropdown menu
drop = tk.OptionMenu(root, clicked, *options)
drop.pack()

# Create Label
label = tk.Label( root , text = " " )

root.mainloop()
