from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    """
    Keywordscraper
    """

    def __init__(self):
        4+5

    def text_prepare(text):

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        NUMBERS = re.compile(r'\d+')
        STOPWORDS = set(stopwords.words('english'))

        text = text.lower()
        # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = REPLACE_BY_SPACE_RE.sub('', text)
        text = BAD_SYMBOLS_RE.sub('', text)
        text = NUMBERS.sub('', text)
        # delete symbols which are in BAD_SYMBOLS_RE from text
        words = text.split()
        i = 0
        while i < len(words):
            if words[i] in STOPWORDS:
                words.pop(i)
            else:
                i += 1
        text = ' '.join(map(str, words))  # delete stopwords from text

        return text

    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(w, 'v') for w in text])

    def basic_preprocessing(df):
        df_temp = df.copy(deep=True)

        try:
            df_temp = df_temp.rename(
                index=str, columns={'transcription': 'text'})
        except:
            pass

        df_temp.loc[:, 'text'] = [Preprocessor.text_prepare(
            x) for x in df_temp['text'].values]
        df_temp["text"] = [
            item for item in df_temp['text'] if not item.isdigit()]

        tokenizer = RegexpTokenizer(r'\w+')

        df_temp["tokens"] = df_temp["text"].apply(tokenizer.tokenize)
        df_temp["lemma"] = df_temp["tokens"].apply(Preprocessor.lemmatize_text)
        df_temp["tokens"] = df_temp["lemma"].apply(tokenizer.tokenize)

        return df_temp
