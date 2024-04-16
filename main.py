import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
import warnings
import matplotlib.pyplot as plt

from aeon.classification.convolution_based import RocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_from_tsfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Function to update the text widget with classifier output
