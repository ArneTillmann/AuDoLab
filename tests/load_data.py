# %load_ext nb_black
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import random
import itertools
from collections import defaultdict


# Preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from itertools import combinations
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# Models

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import OneClassSVM


# Evaluation

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    make_scorer,
)
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import PredefinedSplit, ShuffleSplit


import warnings

warnings.filterwarnings("ignore")

# %matplotlib inline


data = pd.read_csv("tests\mtsamples.csv")
if __name__ == "__main__":
    print(data.shape)
    print(data.columns)
data[data["medical_specialty"] == " Dentistry"]
data = data.sort_values("medical_specialty")
if __name__ == "__main__":
    print(len(data[data["medical_specialty"] == " Dentistry"]))

new_list = list(data[data["medical_specialty"] ==
                " Dentistry"]["transcription"])
if __name__ == "__main__":
    print(len(new_list))

data["dentistry"] = data["transcription"].map(
    lambda x: 1 if x in new_list else -1)
if __name__ == "__main__":
    print(data.shape)
    print(data["dentistry"])

len(data[data["dentistry"] == 1])

data = data.drop_duplicates(
    subset="transcription"
)  # , 'medical_specialty'], keep="first")
if __name__ == "__main__":
    print(data.shape)
data = data.drop(data[data["transcription"].isna()].index)
if __name__ == "__main__":
    print(data.shape)

data = data[["dentistry", "transcription", "medical_specialty"]]
if __name__ == "__main__":
    print(data["medical_specialty"].value_counts().count())

try:
    data = data.reset_index()
except:
    pass

try:
    data = data.drop("level_0", 1)
except:
    pass

try:
    data = data.drop("index", 1)
except:
    pass
if __name__ == "__main__":
    print(data)
