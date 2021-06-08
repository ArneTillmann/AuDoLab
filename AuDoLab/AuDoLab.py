from numpy import round as np_round
from numpy import arange as np_arange
from pandas import concat as pd_concat
from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf
from AuDoLab.subclasses import abstractscraper_arxiv
from AuDoLab.subclasses import abstractscraper_pubmed


class AuDoLab:
    def __init__(self):
        4 + 4

    async def scrape_abstracts(
        self,
        type="arxiv",
        url=None,
        keywords=None,
        operator="OR",
        pages=2,
        in_data="author",
        prepro=False
    ):
        """Function to scrape abstracts of scientific papers from the givin url.
        We used https://ieeexplore.ieee.org/search/advanced to generate a
        list like https://ieeexplore.ieee.org/search/searchresult.jsp?action=se
        arch&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22
        :cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=Tru
        e&rowsPerPage=100&pageNumber=1
        with the search results.
        The abstracts of the papers listet on that list of search results will
        be stored in a .txt file with the givin file name.

        Args:
        - url ( string)
        - file_name (string)
        """
        if type == "ieee":
            ks = abstractscraper.AbstractScraper()
            self.abstracts = await ks.get_abstracts(
                url, keywords, operator, pages, in_data
            )
            

        elif type == "arxiv":
            ks = abstractscraper_arxiv.AbstractScraper_Arxiv()
            self.abstracts = ks.scrape_arxiv(url, pages)
            

        elif type == "pubmed":
            ks = abstractscraper_pubmed.AbstractScraper_Pubmed()
            self.abstracts = ks.scrape_pubmed(url, pages)

        if prepro == True:
            self.abstracts = self.text_cleaning(self.abstracts, "abstract")
        
        return self.abstracts

    def text_cleaning(self, data, column):
        """The data will be lemmatized, tokenized and the stopwords will be
        deleted.

        Arguments:
        - data (<class 'pandas.core.frame.DataFrame'>)
        """
        prepro = preprocessing.Preprocessor()
        self.data_processed = prepro.basic_preprocessing(data, column)
        return self.data_processed

    def tf_idf(self, data, papers, data_column, papers_column, features=None):
        """The function tf_idf(...) calculates the tfidf scores.

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
        num_topics,
        corpus=None,
        dict=None,
        no_below=None,
        no_above=None,
        random_state=101,
        passes=20,
        chunksize=500,
        eta="auto",
        eval_every=None,
        multi=True,
        alpha="asymmetric",
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
        if corpus == None:
            self.dictionary, self.bow_corpus = lda.LDA.preperation(
                data, no_below, no_above
            )

        else:
            self.corpus = (corpus,)
            self.dictionary = dict

        self.lda = lda.LDA()
        self.lda_model = self.lda.model(
            corpus=self.bow_corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            random_state=random_state,
            passes=passes,
            chunksize=chunksize,
            eta=eta,
            eval_every=eval_every,
            multi=multi,
            alpha=alpha,
        )
        return self.lda_model

    def lda_visualize_topics(
        self,
        save_name="audolab_model.png",
        lda_model=None,
        bow_corpus=None,
        dictionary=None,
        type="pyldavis",
        figsize=(20, 10),
        facecolor="k",
        width=1600,
        height=800,
        background_color="white",
        topic=0,
        words=100,
        save=False,
        n_clouds=1,
    ):
        """The lda model calculated with the function lda_modeling is visualized
        in an html frame and opened in the standard browser.
        """
        if bow_corpus == None:
            bow_corpus = self.bow_corpus

        if dictionary == None:
            dictionary = self.dictionary

        if lda_model == None:
            lda_model = self.lda_model

        lda.LDA.visualize_topics(
            lda_model,
            bow_corpus=bow_corpus,
            dictionary=dictionary,
            save_name=save_name,
            type=type,
            figsize=figsize,
            facecolor=facecolor,
            width=width,
            height=height,
            background_color=background_color,
            topic=topic,
            words=words,
            save=save,
            n_clouds=n_clouds,
        )
