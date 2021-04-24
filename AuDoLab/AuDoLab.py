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

    async def scrape_abstracts(self, url=None, keywords=None, operator="OR",
                               pages=2, in_data="author"):
        """
        Function to scrap abstracts of scientific papers from the givin url.
        We used https://ieeexplore.ieee.org/search/advanced to generate a
        list like https://ieeexplore.ieee.org/search/searchresult.jsp?action=se
        arch&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22
        :cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=Tru
        e&rowsPerPage=100&pageNumber=1
        with the search results.
        The abstracts of the papers listet on that list of search results will
        be stored in a .txt file with the givin file name.

        :param url: when the user specifies an own search query on IEEEXplore,
            defaults to None
        :type url: string
        :param keywords: keywords that are searched for, defaults to     None
        :type keywords: list of strings
        :param operator: "and" / "or" operator between keywords, defaults to "OR"
        :type operator: string
        :param pages: number of pages that is iterated over, defaults to 2
        :type pages: int
        :param in_data: "author" or "all_meta" whether to search in author
                keywords or all metadata, defaults to "author"
        :type in_data: string

        :return: DataFrame with the scraped abstracs in column="text"
        :rtype: pd.DataFrame
        """

        ks = abstractscraper.AbstractScraper()

        self.abstracts = await ks.get_abstracts(url, keywords, operator, pages, in_data)
        print(self.abstracts)
        #file_name = file_name + ".txt"
        #self.abstracts.to_csv(self.abstracts, header=True, index=False)
        return self.abstracts

    def preprocessing(self, data, column):
        """Preprocessing function that calls the helper functions

        :param data: DataFrame that has the text data stored
        :type data: pd.DataFrame
        :param column: column name where raw text is stored
        :type column: str

        :return: DataFrame with preprocessed text
        :rtype:  DataFrame
        """
        prepro = preprocessing.Preprocessor()
        self.data_processed = prepro.basic_preprocessing(
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

    def tf_idf(self, data, papers, data_column="lemma", papers_column="lemma",
               features=None, ngrams=2):
        """creates tf-idf objects for one-class SVM classification. The tf-idf
        scores are calculated over a joint corpus,
        however, the target data and the out-of-domain training data are stored
        in seperate, as the one-class SVM is only trained on
        the tf-idf scores of the out-of-domain training data

        :param data: preprocessed target documents
        :type: DataFrame
        :param papers: preprocessed out-of-domain training data
        :type: DataFrame
        :param data_colum: name of columnin target dataframe where
            lemmatized documents are stored, defaults to 'lemma'
        :type: String
        :param papers_colum: name of column in out-of-domain training
            dataframe where lemmatized documents are stored, defaults to 'lemma'
        :type: String
        :param ngrams: whether ngram are formed, defaults to 2
        :type: int
        :param features: number of max features, defaults to 8000.
        :type: int

        :return: tfidf object data for target data and ou-of-domain training
            data
        :type: data and papers
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
        nus=np_round(np_arange(0.1, 0.5, 0.01), 5),
        quality_train=0.85,
        min_pred=0.05,
        max_pred=0.2,
    ):
        """
        trains a one-class SVM on the out-of-domain training data


        :param training: training dataset of preprocessed documents
        :type training: pd.DataFrame
        :param predicting: target dataset of preprccessed documents
        :type predicting: pd.DataFrame
        :param nus: hyperparameters over which are looped. For each nu
            the classifiers is trained
        :type nus: list of floats
        :param quality_train: percentage of training data that seems to
            belong to target class, defaults to 0.85
        :type quality_train: float
        :param min_pred: percentage of target data that has to be at
            least classified as belonging to target class
            for classifier to be considered ,defaults to 0.05
        :type min_pred: float
        :param max_pred: percentage of target class that is maximally
            allowed to be classified as belonging to
            target class for classifier to be considered, defaults to 0.2
        :type max_pred: float

        :return: DataFrame with stored classifiers that
            fulfill conditions
        :rtype: pd.DataFrame
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
        """
        returns dataframe where documents that are classified to target class
        have 1, otherwise, 0

        :param df: dataframe of target documents
        :type df: pd.Dataframe
        :param classifier: list of all possible o-svm classifiers
        :type classifier: list
        :param i: index of which classifier is chosen/preferred
        :type i: int

        :return: documents that are classified as belonging to target
        :rtype: pd.dataframe
            class by o-svm
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
        """LDA model

        :param data: DataFrame stat store tokenized text in col = "tokens"
        :type data: pd.DataFrame
        :param no_below: if word appears in less than no_below documents they
            are not considered, defaults to None
        :type no_below: float
        :param no_above: if e.g. 0.9, no words are taken into account that
            appear more often than 90%, defaults to None
        :type no_above: float
        :param num_topics: pre-defined number of topics
        :type num_topics: int
        :param random_state: for recreating exact identical output
        :type random_state: int
        :param passes: Number of passes through the corpus during training.
        :type passes: int

        :return: returns lda_model output
        :rtype: lda_model
        """

        self.dictionary, self.bow_corpus = lda.LDA._preperation(data, no_below,
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
