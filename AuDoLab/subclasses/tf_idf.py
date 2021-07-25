from AuDoLab.subclasses.preprocessing import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer


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
        """creates tf-idf objects for one-class SVM classification. The tf-idf
            scores are calculated over a joint corpus,
            however, the target data and the out-of-domain training data are
            stored in seperate, as the one-class SVM is only trained on
            the tf-idf scores of the out-of-domain training data

        :param data: preprocessed target documents
        :type data: DataFrame
        :param papers: preprocessed out-of-domain training data
        :type papers: DataFrame
        :param data_colum: name of columnin target dataframe where
            lemmatized documents are stored, defaults to 'lemma'
        :type data_colum: String
        :param papers_colum: name of column in out-of-domain training
            dataframe where lemmatized documents are stored, defaults to 'lemma'
        :type papers_colum: String
        :param ngrams: whether ngram are formed, defaults to 2
        :type ngrams: int
        :param features: number of max features, defaults to 8000.
        :type features: int

        :return: tfidf object data for target data and ou-of-domain training
            data
        :type: data and papers
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
