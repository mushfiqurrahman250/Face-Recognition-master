import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_csv("D:/untitled/Face-Recognition-master/face.csv")
data.info()
data.describe()
data.head()

