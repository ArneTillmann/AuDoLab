from numpy import round as np_round
from numpy import arange as np_arange
from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf


class AuDoLab:
    def __init__(self):
        4 + 4

    def scrape_abstracts(self, url, file_name):
        """Function to scrap abstracts of scientific papers from the givin url.
        We used https://ieeexplore.ieee.org/search/advanced to generate a
        list like https://ieeexplore.ieee.org/search/searchresult.jsp?action=se
        arch&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22
        :cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=Tru
        e&rowsPerPage=100&pageNumber=1
        with the search results.
        The abstracts of the papers listet on that list of search results will
        be stored in a .txt file with the givin file name.

        Arguments:
        - url ( string)
        - file_name (string)
        """

        ks = abstractscraper.AbstractScraper(url)

        ks.open()
        ks.find_links()
        self.abstracts = ks.get_abstracts()
        file_name = file_name + ".txt"
        self.abstracts.to_csv(file_name, header=True, index=False)
        return self.abstracts

    def preprocessing(self, data, column):
        """ The data will be lemmatized, tokenized and the stopwords will be
        deleted.

        Arguments:
        - data (<class 'pandas.core.frame.DataFrame'>)
        """
        self.data_processed = preprocessing.Preprocessor.basic_preprocessing(
            data, column
        )
        return self.data_processed

    # def tf_idf_features(self, data, papers, features=8000):
    #     """The function tf_idf_features(...) calculates the tfidf scores, but
    #     return only the <features> amount of words with the highest tfidf
    #     scores.
    #
    #     Arguments:
    #     - data (<class 'pandas.core.frame.DataFrame'>)
    #     - papers (<class 'pandas.core.frame.DataFrame'>)
    #     """
    #     tfidf = tf_idf.Tf_idf()
    #     self.data_tfidf_features, self.papers_tfidf_features =
    #     tfidf.tfidf_features(
    #         data, papers, features=features)
    #     return self.data_tfidf_features, self.papers_tfidf_features

    def tf_idf(self, data, papers, data_column, papers_column, features=None):
        """ The function tf_idf(...) calculates the tfidf scores.

        Arguments:
        - data (<class 'pandas.core.frame.DataFrame'>)
        - papers (<class 'pandas.core.frame.DataFrame'>)
        """
        tfidf = tf_idf.Tf_idf()
        self.data_tfidf, self.papers_tfidf = tfidf.tfidf(
            data, papers, data_column, papers_column, features
        )
        return self.data_tfidf, self.papers_tfidf

    def one_class_svm(
        self,
        training,
        predicting,
        nus=np_round(np_arange(0.001, 0.5, 0.001), 5),
        quality_train=0.85,
        min_pred=0.01,
        max_pred=0.1,
    ):
        """
        This is a one class classifier, that uses the training data (usually the
        papers we scraped earlier) to classify the predicting data. The
        returned df might contain multiple classifier for different parameters
        nu(s).

        Arguments:
        - training (<class 'scipy.sparse.csr.csr_matrix'>)
        - predicting (<class 'scipy.sparse.csr.csr_matrix'>)
        - nus (np.array)
        - quality_train (float)
        - min_pred (float)
        - max_pred (float)
        """
        one_Class_SVM = one_class_svm.One_Class_SVM()
        self.df = one_Class_SVM.classification(
            training=training,
            predicting=predicting,
            nus=nus,
            quality_train=quality_train,
            min_pred=min_pred,
            max_pred=max_pred,
        )
        return self.df

    def choose_classifier(self, df, classifier, i):
        """As mentioned in the description of the function one_class_svm(...)
        multiple classifiers might be return. Now you choose one with which
        you want to continue. The df will then only continue the entries that
        were positivly classified.

        Arguments:
        - df (<class 'pandas.core.frame.DataFrame'>)
        - classifier (<class 'pandas.core.frame.DataFrame'>)
        - i (int)
        """
        return one_class_svm.One_Class_SVM.choose_classifier(df, classifier, i)

    def lda_modeling(
        self,
        data,
        no_below=None,
        no_above=None,
        num_topics=5,
        random_state=101,
        passes=20,
    ):
        """The function performs lda modelling as described in this
        https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf paper.

        Arguments:
        - data (<class 'pandas.core.frame.DataFrame'>)
        - no_below (int)
        - no_above (int)
        - num_topics (int)
        - random_state (int)
        - passes (int)
        """

        self.dictionary, self.bow_corpus = lda.LDA.preperation(data, no_below,
                                                               no_above)
        self.lda = lda.LDA()
        self.lda_model = self.lda.model(
            self.bow_corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            random_state=random_state,
            passes=passes,
        )
        return self.lda_model

    def lda_visualize_topics(self):
        """The lda model calculated with the function lda_modeling is visualized
        in an html frame and opened in the standard browser.
        """
        lda.LDA.visualize_topics(self.lda_model, self.bow_corpus,
                                 self.dictionary)
