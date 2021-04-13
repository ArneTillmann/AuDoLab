# %load_ext nb_black
import pandas as pd
import numpy as np


# %matplotlib inline


data = pd.read_csv("tests\mtsamples.csv")
data = data.sort_values("medical_specialty")

new_list = list(data[data["medical_specialty"] ==
                " Dentistry"]["transcription"])


data["dentistry"] = data["transcription"].map(
    lambda x: 1 if x in new_list else -1)
data = data.drop_duplicates(
    subset="transcription"
)  # , 'medical_specialty'], keep="first")


data = data.drop(data[data["transcription"].isna()].index)

data = data[["dentistry", "transcription", "medical_specialty"]]

data=data[["transcription"]]
if __name__ == "__main__":
    print(data)
