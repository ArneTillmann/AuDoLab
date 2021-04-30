# %load_ext nb_black
import pandas as pd


data = pd.read_csv(r"tests/mtsamples.csv")

data[data["medical_specialty"] == " Dentistry"]

data = data.sort_values("medical_specialty")


new_list = list(
    data[data["medical_specialty"] == "Dentistry"]["transcription"]
)


data["dentistry"] = data["transcription"].map(
    lambda x: 1 if x in new_list else -1
)
data = data.drop_duplicates(
    subset="transcription"
)  # , 'medical_specialty'], keep="first")



data = data.drop(data[data["transcription"].isna()].index)

data = data[["dentistry", "transcription", "medical_specialty"]]

data = data[["transcription"]]
data = data.reset_index(drop=True)
if __name__ == "__main__":
    print(data.reset_index(drop=True))
