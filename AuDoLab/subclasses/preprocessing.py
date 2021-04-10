from re import compile as re_compile
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        4 + 5

    def _text_prepare(text, language="english"):
        """prepares text for lda and o-svm.
        Removes stopwords, symbols, double spaces and numbers

        Args:
            text (numpy array): [column out of dataframe with documents]
            language (string): [language for stopword removel, default: english]

        Returns:
            [numpy array]: [prepared documents]
        """

        REPLACE_BY_SPACE_RE = re_compile("[/(){}\[\]\|@,;]")
        BAD_SYMBOLS_RE = re_compile("[^0-9a-z #+_]")
        NUMBERS = re_compile(r"\d+")
        STOPWORDS = set(stopwords.words(language))

        text = text.lower()
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub("", text)
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

    def _lemmatize_text(text):
        """helper function

        Args:
            text (numpy array): [dataframe column where documents are stored]

        Returns:
            [numpy array]: [lemmatized documents]
        """
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(w, "v") for w in text])

    def basic_preprocessing(df, column):
        """calls helper functions from above and preprocesses documents
        removes all stopwords and unnecessary terms, e.g. symbols, numbers, double spaces
        and tokenizes documents

        Args:
            df (DataFrame): [dataframe where documents are stored in one column]
            columns (String): [column name where documents are stored]

        Returns:
            [DataFrame]: [dataframe with preprocessed documents. 2 new columns are appended. The completely preprocessed
            documents are stored in column ['tokens'], the only lemmatized documents are stored in columns ['lemma']]
        """
        df_temp = df.copy(deep=True)

        df_temp.loc[:, column] = [
            Preprocessor._text_prepare(x, "english") for x in df_temp[column].values
        ]
        df_temp[column] = [item for item in df_temp[column] if not item.isdigit()]

        tokenizer = RegexpTokenizer(r"\w+")

        df_temp["tokens"] = df_temp[column].apply(tokenizer.tokenize)
        df_temp["lemma"] = df_temp["tokens"].apply(Preprocessor._lemmatize_text)
        df_temp["tokens"] = df_temp["lemma"].apply(tokenizer.tokenize)

        return df_temp
