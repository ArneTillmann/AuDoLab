import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Text preparation
class Preprocessor:
    def __init__(self):
        4+5 

    def text_prepare(self, text, language="english"):
        """text preparation function for text preprocessing

        Args:
            text (helper variable): None
            language (str, optional): [description]. Defaults to "english". Sets language of stopwords to be removed
        """

        REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
        BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
        NUMBERS = re.compile("\d+")
        STOPWORDS = set(stopwords.words(language))

        text = text.lower()
        text = REPLACE_BY_SPACE_RE.sub(
            "", text
        )  # replace REPLACE_BY_SPACE_RE symbols by space in text
        
        text = BAD_SYMBOLS_RE.sub("", text)
        text = NUMBERS.sub("", text)
        # delete symbols which are in BAD_SYMBOLS_RE from text
        words = text.split()
        i = 0
        while i < len(words):
            if words[i] in STOPWORDS:
                words.pop(i)
            else:
                i += 1
        text = " ".join(map(str, words))  # delete stopwords from text

        return text

    def _lemmatize_text(self, text):
        """helper function that lemmatizes already tokenized text
        """

        lemmatizer = nltk.stem.WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(w, "v") for w in text])

    def basic_preprocessing(self, df, column):
        """Preprocessing function that calls the helper functions

        Args:
            df (DataFrame): DataFrame that has the text data stored
            column (str): column name where raw text is stored

        Returns:
            DataFrame: DataFrame with preprocessed text
        """
        
        df_temp = df.copy(deep=True)
        df_temp[column] = df_temp[column].astype(str)

        df_temp.loc[:, column] = [self.text_prepare(x) for x in df_temp[column].values]
        df_temp[column] = [item for item in df_temp[column] if not item.isdigit()]

        tokenizer = RegexpTokenizer(r"\w+")

        df_temp["tokens"] = df_temp[column].apply(tokenizer.tokenize)
        df_temp["lemma"] = df_temp["tokens"].apply(self._lemmatize_text)
        df_temp["tokens"] = df_temp["lemma"].apply(tokenizer.tokenize)

        return df_temp


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("mtsamples.csv")
    prepro = Preprocessor()
    test = prepro.basic_preprocessing(df=data, column="transcription")
