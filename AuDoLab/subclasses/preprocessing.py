import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import re
    import nltk
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from gensim.models import Phrases
    from tqdm import tqdm


# Text preparation


class Preprocessor:
    def __init__(self):
        4 + 5

    def _text_prepare(self, text, language="english", stop_words=True):
        """text preparation function for text preprocessing

        Args:
            text (helper variable): None

            language (str, optional): Sets language of stopwords to be removed.
                Defaults to "english".

            stop_words (bool, optional): If true, the stopwords are removed, if
                not, stopwords are left as they are. Defaults to True.

        Returns:
            [type]: [description]
        """

        REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
        BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
        NUMBERS = re.compile(r"\d+")

        text = text.lower()
        text = REPLACE_BY_SPACE_RE.sub(
            "", text
        )  # replace REPLACE_BY_SPACE_RE symbols by space in text

        text = BAD_SYMBOLS_RE.sub("", text)
        text = NUMBERS.sub("", text)
        # delete symbols which are in BAD_SYMBOLS_RE from text
        words = text.split()

        if stop_words:
            i = 0
            STOPWORDS = set(stopwords.words(language))
            while i < len(words):
                if words[i] in STOPWORDS:
                    words.pop(i)
                else:
                    i += 1
        text = " ".join(map(str, words))  # delete stopwords from text

        return text

    def _lemmatize_text(self, text):
        """helper function that lemmatizes already tokenized text"""

        lemmatizer = nltk.stem.WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(w, "v") for w in text])

    def _preprocessing(self, df, column):
        """Preprocessing function that calls the helper functions

        :param df: DataFrame that has the text data stored
        :type df: pd.DataFrame
        :param column: column name where raw text is stored
        :type column: str

        :return: DataFrame with preprocessed text
        :rtype:  DataFrame
        """

        df_temp = df.copy(deep=True)
        df_temp[column] = df_temp[column].astype(str)

        df_temp.loc[:, column] = [self._text_prepare(
            x) for x in df_temp[column].values]
        df_temp[column] = [
            item for item in df_temp[column] if not item.isdigit()]

        tokenizer = RegexpTokenizer(r"\w+")

        df_temp["tokens"] = df_temp[column].apply(tokenizer.tokenize)
        df_temp["lemma"] = df_temp["tokens"].apply(self._lemmatize_text)
        df_temp["tokens"] = df_temp["lemma"].apply(tokenizer.tokenize)

        return df_temp, df_temp["tokens"]

        # apply prepro func

    def basic_preprocessing(self, df, column, ngram_type=2):
        """The data will be lemmatized, tokenized and the stopwords will be
        deleted.

        Args:
            df (pd.DataFrame): Dataframe where the documents to be preprocessed
                are stored.

            column (str):  Column name of the column where docs are stored.

            ngram_type (int, optional): Number of ngrams used. Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame where the original docus and the
                preprocessed documents are stored.
        """
        df, df_txt = self._preprocessing(df=df, column=column)
        df_txt = df_txt.reset_index()
        df_txt = df_txt.drop("index", axis=1)
        df_txt = df_txt["tokens"]

        df = df.reset_index()
        df = df.drop("index", axis=1)

        if ngram_type == 2:
            bigram = Phrases(df_txt, min_count=10)

            for idx in tqdm(range(len(df_txt))):

                for token in bigram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)

        if ngram_type == 3:
            bigram = Phrases(df_txt, min_count=10)
            trigram = Phrases(bigram[df_txt])

            for idx in tqdm(range(len(df_txt))):
                for token in bigram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)
                for token in trigram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)

        if ngram_type == 4:
            bigram = Phrases(df_txt, min_count=10)
            trigram = Phrases(bigram[df_txt])
            fourgram = Phrases(trigram[df_txt])

            for idx in tqdm(range(len(df_txt))):
                for token in bigram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)
                for token in trigram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)
                for token in fourgram[df_txt[idx]]:
                    if "_" in token:
                        df_txt[idx].append(token)

        if ngram_type >= 4:
            print("please specify a ngram_type <= 4")

        df["preprocessed"] = df_txt
        df["lemma"] = [" ".join(map(str, j)) for j in df["preprocessed"]]

        df = df.drop(["tokens"], axis=1)

        return df


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("mtsamples.csv")
    prepro = Preprocessor()
    test = prepro.basic_preprocessing(
        df=data, column="transcription", ngram_type=3)
