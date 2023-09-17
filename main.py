import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree


print("Dataset:")
dataset = pd.read_csv('')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
pyplot.show()

## Two Variables
A = dataset[dataset.Result == 1]
B = dataset[dataset.Result == 0]

plt.scatter(A.age,A.Smokes)