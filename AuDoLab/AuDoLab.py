import pandas as pd
from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf
from AuDoLab.subclasses import abstractscraper_arxiv
from AuDoLab.subclasses import abstractscraper_pubmed
import sys


class AuDoLab:
    """TODO
        1) call asyncio func directly in class such that the user does not need to call advanced loop
        2) Remove depreciationwarnings (scipy.sparse and imp)
        3) include pyldavis function (or remove completely depending on utility for users)
        4) adapt documentation that it is consistent within itself (documentation = summarys and datatypes)
        5) check if further functions needed -> second loop of iteration in decision rule from (https://www.researchgate.net/profile/Astrid-Krenz/publication/344432349_Unsupervised_Document_Classification_integrating_Web_Scraping_One-Class_SVM_and_LDA_Topic_Modelling/links/5f749c97458515b7cf596d36/Unsupervised-Document-Classification-integrating-Web-Scraping-One-Class-SVM-and-LDA-Topic-Modelling.pdf)

    """
    def __init__(self):
        4 + 4

    async def ieee_scraper(
        self,
        url=None,
        keywords=None,
        operator="OR",
        pages=2,
        in_data="author",
        prepro=False,
        ngram_type=2,
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
            url (str, optional): [description]. Defaults to None.
            keywords (list, optional): List of keywords that are searched for. Defaults to None.
            operator (str, optional): Operator between the keywords. "AND" or "OR". If "AND" the search results must include all keywords. Defaults to "OR".
            pages (int, optional): Number of pages that are iterated over. Translates directly to number of abstracts that are scraped. Roughly there are 100 abstracts scraped per page. Defaults to 2.
            in_data (str, optional): If the keywords are searched for in the author keywords or in all metadata. Defaults to "author".
            prepro (bool, optional): if True, the scraped data will directly be preprocessed for later use. Defaults to False.
            ngram_type (int, optional): number of ngrams in preprocessing. Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame with the stored abstracts and metadata
        """

        number = pages

        ks = abstractscraper.AbstractScraper()
        self.abstracts = await ks.get_abstracts(
            url=url, keywords=keywords, operator=operator, pages=number, in_data=in_data
        )

        if prepro == True:
            self.abstracts = self.abstracts.reset_index(drop=True)
            self.abstracts = self.text_cleaning(
                self.abstracts, "abstract", ngram_type=ngram_type
            )

        if type(self.abstracts) != pd.DataFrame:
            print(
                "if using the ieee abstractscraper, please use the following code: \n \n"
                + "async def scrape():"
                + "\n     return await audo.ieee_scraper(keywords=[keywords], prepro=False, pages=1, ngram_type=2)"
                + "\n\nscraped_documents = asyncio.get_event_loop().run_until_complete(scrape())"
            )

            sys.exit(
                "please specify the code as indicated above, or use the function abstract_scraper to scrape from different websites"
            )

        return self.abstracts

    def abstract_scraper(
        self, type="arxiv", url=None, pages=2, prepro=False, ngram_type=2
    ):
        """Scapes the pages arxiv.org and pubmed.gov for paper abstracts

        Args:
            type (str, optional): "arxiv" or "pubmed". Defines for which page the scraping is done. Defaults to "arxiv".
            url (str, optional): The given url after which the papers are scraped. Must be in line with type. Defaults to None.
            pages (int, optional): Number of pages that are iterated over. Defaults to 2.
            prepro (bool, optional): If True, the scraped documents are preprocessed directly. Defaults to False.
            ngram_type (int, optional): Number of ngrams in preprocessing. Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame with the stored abstracts
        """

        if type == "arxiv":
            ks = abstractscraper_arxiv.AbstractScraper_Arxiv()
            self.abstracts = ks.scrape_arxiv(url, pages)

        elif type == "pubmed":
            ks = abstractscraper_pubmed.AbstractScraper_Pubmed()
            self.abstracts = ks.scrape_pubmed(url, pages)

        self.abstracts = self.abstracts.reset_index(drop=True)

        if prepro == True:
            self.abstracts = self.text_cleaning(
                self.abstracts, "abstract", ngram_type=ngram_type
            )

        return self.abstracts

    def text_cleaning(self, data, column, ngram_type=2):
        """The data will be lemmatized, tokenized and the stopwords will be
        deleted.

        Args:
            data (pd.DataFrame): Dataframe where the documents to be preprocessed are stored
            column (str): Column name of the column where docs are stored
            ngram_type (int, optional): Number of ngrams used. Defaults to 2.

        Returns:
            pd.DataFrame: DataFrame where the original docus and the preprocessed documents are stored
        """

        prepro = preprocessing.Preprocessor()
        self.data_processed = prepro.basic_preprocessing(
            data, column, ngram_type=ngram_type
        )
        return self.data_processed

    def tf_idf(self, data, papers, data_column, papers_column, features=None, ngrams=2):
        """[creates tf-idf objects for one-class SVM classification. The tf-idf
        scores are calculated over a joint corpus,
        however, the target data and the out-of-domain training data are stored
        in seperate, as the one-class SVM is only trained on
        the tf-idf scores of the out-of-domain training data]

        Args:
            data (DataFrame): [preprocessed target documents]
            papers (DataFrame): [preprocessed out-of-domain training data]
            data_colum (String): [name of columnin target dataframe where
            lemmatized documents are stored]. Defaults to 'lemma'
            papers_colum (String): [name of column in out-of-domain training
            dataframe where lemmatized documents are stored]. Defaults to
            'lemma' ngrams (int, optional): [whether ngram are formed].
            Defaults to 2.
            features (int, optional): [number of max features].
            Defaults to 8000.

        Returns:
            [data and papers]: [tfidf object data for target data and
             ou-of-domain training data]
        """

        tfidf = tf_idf.Tf_idf()
        self.data_tfidf, self.papers_tfidf = tfidf.tfidf(
            data, papers, data_column, papers_column, features, ngrams
        )
        return self.data_tfidf, self.papers_tfidf

    def one_class_svm(
        self,
        training,
        predicting,
        nus,
        quality_train=0.85,
        min_pred=0.05,
        max_pred=0.2,
        gamma="auto",
        kernel="rbf",
    ):
        """[summary]

        Args:
            training (DataFrame): training dataset of preprocessed documents
            predicting (DataFrame): target dataset of preprccessed documents
            nus (list of floats): hyperparameters over which are looped. For each nu the classifier is trained
            quality_train (float, optional): percentage of training data that seems to
                                    belong to target class. Default: 0.85. Defaults to 0.85.
            min_pred (float, optional): percentage of target data that has to be at
                               least classified as belonging to target class
            for classifier to be considered. Default: 0.0. Defaults to 0.05.
            max_pred (float, optional): percentage of target class that is maximally
                               allowed to be classified as belonging to
            target class for classifier to be considered.. Defaults to 0.2.
            gamma (str, optional): Hyperparamter of O-SVM. Defaults to "auto".
            kernel (str, optional): Kernel function used in O_SVM. Defaults to "rbf".

        Returns:
            pd.DataFrame: DataFrame with stored classifiers that fulfill conditions
        """

        one_Class_SVM = one_class_svm.One_Class_SVM()
        self.df = one_Class_SVM.classification(
            training=training,
            predicting=predicting,
            nus=nus,
            quality_train=quality_train,
            min_pred=min_pred,
            max_pred=max_pred,
            gamma=gamma,
            kernel=kernel,
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
        column="preprocessed",
    ):
        """The function performs lda modelling as described in this
        https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf paper.

        Args:
            corpus (iterable of list of (int, float), optional): Stream of document vectors or sparse matrix of shape
            num_topics (int): pre-defined number of topics]
            id2word ({dict of (int, str): gensim.corpora.dictionary.Dictionary}) –Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for debugging and topic printing.
            random_state (int): for recreating exact identical output. Defaults to 101.
            passes (int): Number of passes through the corpus during training.Defaults to 20.
            chunksize (int, optional): chunksize in lda passes. Defaults to 500.
            eta (str, optional): [description]. Defaults to "auto".
            eval_every ([type], optional): Hyperparameter in LDA used to initiliaze the Dirichlet distribution. Defaults to None.
            multi (bool, optional): If true, the in gensim incorporated multicore variant is used. Defaults to True.
            alpha (str, optional): OTher Dirichlet Prior. Defaults to "asymmetric".

        Returns:
            [lda_model]: [returns lda_model output]
        """

        if corpus == None:
            self.dictionary, self.bow_corpus = lda.LDA.preperation(
                data, no_below, no_above, column=column
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
        """Visualizes the topic models output in wordclouds or pyldavis

        TODO: pyldavis not yet working, especially not in jupyter notebooks. MAybe take out all together

        Args:
            lda_model (gensim.models.ldamodel.LdaModel): the created LDA model
            bow_corpus (gensim.corpora.dictionary.Dictionary): Bag of words corpus of used documents
            dictionary (gensim.corpora.dictionary.Dictionary): Dictionary of all words
            save_name (str, optional): name under which the plots should be save. Defaults to "audolab_model.png".
            type (str, optional): type of visualisation- either "clouds" or "pyldavis". Defaults to "clouds".
            figsize (tuple, optional): Size of wordclouds. Defaults to (50, 30).
            facecolor (str, optional): Colour of wordcloud Defaults to "k".
            width (int, optional): width of plots. Defaults to 2000.
            height (int, optional): height of plots. Defaults to 1000.
            background_color (str, optional): Background colour of wordcloud. Defaults to "white".
            topic (int, optional): IF only one wordcloud is plotted, index of topic that is plotted. Defaults to 0.
            words (int, optional): Number of words per cloud. Defaults to 100.
            save (bool, optional): whether the plots should be saved or not. Defaults to False.
            n_clouds (int, optional): Number of word clouds that are plotted. Defaults to 1.

        Raises:
            ValueError: If save_name is not a string: no "Please specify a string as the name under which the plots should be saved"

        Returns:
            None:
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
