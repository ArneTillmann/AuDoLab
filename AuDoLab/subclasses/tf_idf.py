import warnings
from sklearn.feature_extraction.text import TfidfVectorizer

def warn(*args, **kwargs):
    pass

warnings.warn = warn

class Tf_idf:

    def __init__(self):
        4 + 5

    @staticmethod
    def tfidf(
        data,
        papers,
        data_column="lemma",
        papers_column="lemma",
        features=None,
        ngrams=2,
    ):
        """Creates tf-idf objects for one-class SVM classification.

        The tf-idf scores are calculated over a joint corpus, however the target
        data and the out-of-domain training data are stored in seperate, as the
        one-class SVM is only trained on the tf-idf scores of the out-of-domain
        training data.

        Args:
            data (DataFrame): preprocessed target documents

            papers (DataFrame): preprocessed out-of-domain training data

            data_colum (String): name of columnin target dataframe where
                lemmatized documents are stored. Defaults to 'lemma'

            papers_colum (String): name of column in out-of-domain training
                dataframe where lemmatized documents are stored. Defaults to
                'lemma'

            ngrams (int, optional): whether ngram are formed.
                Defaults to 2.

            features (int, optional): number of max features.
                Defaults to 8000.

        Returns:
            data and papers: tfidf object data for target data and
                out-of-domain training data
        """

        df_temp = data.copy(deep=True)
        papers_temp = papers.copy(deep=True)

        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, ngrams), max_features=features
        )

        corpus = df_temp[data_column].tolist(
        ) + papers_temp[papers_column].tolist()
        tfidf_vectorizer.fit(corpus)

        data_corpus = df_temp[data_column].tolist()
        paper_corpus = papers_temp[papers_column].tolist()

        data = tfidf_vectorizer.transform(data_corpus)
        papers = tfidf_vectorizer.transform(paper_corpus)

        return data, papers
