# %load_ext nb_black
import pandas as pd
import numpy as np


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
data=data[["transcription"]]
if __name__ == "__main__":
    print(data)
