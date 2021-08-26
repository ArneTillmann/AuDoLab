import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import webbrowser
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from gensim import corpora, models
    import pyLDAvis.gensim_models
    import pyLDAvis


def warn(*args, **kwargs):
    pass


warnings.warn = warn

warnings.filterwarnings("ignore", category=DeprecationWarning)


class LDA:
    """
    performs Latent Dirichlet Allocation Topic Modelling
    """

    def __init__(self):
        pass

    @staticmethod
    def preperation(df_processed, no_below=None, no_above=None, column="preprocessed"):
        """Preprocessing for LDA

        Args:
            df_processed (pd.DataFrame): DataFrame that stores tokenized text in
                col ["tokens"]

            no_below (int): if word appears in less than
                no_below documents they are not considered. Defaults to None.

            no_above (float): if e.g. 0.9, no words are taken into
                account that appear more often than 90%. Defaults to None.

        Returns:
            dictionary: [returns the dictionary used for LDA]

            bow_corpus: [Corpus used for LDA]
        """

        dictionary = corpora.Dictionary(df_processed[column])
        if no_below is None:
            no_below = 0
        if no_above is None:
            no_above = 1
        bow_corpus = [dictionary.doc2bow(doc) for doc in df_processed[column]]
        return dictionary, bow_corpus

    @staticmethod
    def model(
        corpus,
        num_topics,
        id2word,
        random_state=101,
        passes=20,
        chunksize=500,
        eta="auto",
        eval_every=None,
        multi=True,
        alpha="asymmetric",
    ):
        """LDA model

        Args:
            corpus (iterable of list of (int, float), optional): Stream of
                document vectors or sparse matrix of shape

            num_topics (int): pre-defined number of topics

            id2word ({dict of (int, str):
                gensim.corpora.dictionary.Dictionary}) –Mapping from word IDs to
                words. It is used to determine the vocabulary size, as well as
                for debugging and topic printing.

            random_state (int): for recreating exact identical output. Defaults
                to 101.

            passes (int): Number of passes through the corpus during training.
                Defaults to 20.

            chunksize (int, optional): chunksize in lda passes. Defaults to 500.

            eta (str, optional): [description]. Defaults to "auto".

            eval_every ([type], optional): Hyperparameter in LDA used to
                initiliaze the Dirichlet distribution. Defaults to None.

            multi (bool, optional): If true, the in gensim incorporated
                multicore variant is used. Defaults to True.

            alpha (str, optional): OTher Dirichlet Prior. Defaults to
                "asymmetric".

        Returns:
            lda_model: returns lda_model output
        """

        if multi is True:
            lda_model = models.LdaMulticore(
                random_state=random_state,
                alpha=alpha,
                corpus=corpus,
                num_topics=num_topics,
                id2word=id2word,
                passes=passes,
                chunksize=chunksize,
                eta=eta,
                eval_every=eval_every,
            )

        else:
            lda_model = models.LdaModel(
                random_state=random_state,
                alpha=alpha,
                corpus=corpus,
                num_topics=num_topics,
                id2word=id2word,
                passes=passes,
                chunksize=chunksize,
                eta=eta,
                eval_every=eval_every,
            )

        return lda_model

    @staticmethod
    def visualize_topics(
        lda_model,
        bow_corpus,
        dictionary,
        save_name="audolab_model.png",
        type="clouds",
        figsize=(50, 50),
        facecolor="k",
        width=2000,
        height=1000,
        background_color="white",
        topic=0,
        words=100,
        save=False,
        n_clouds=1,
    ):
        """Visualizes the topic models output in wordclouds or pyldavis

        Args:
            lda_model (gensim_models.ldamodel.LdaModel): the created LDA model

            bow_corpus (gensim.corpora.dictionary.Dictionary): Bag of words
                corpus of used documents

            dictionary (gensim.corpora.dictionary.Dictionary): Dictionary of
                all words

            save_name (str, optional): name under which the plots should be
                save. Defaults to "audolab_model.png".

            type (str, optional): type of visualisation- either "clouds" or
                "pyldavis". Defaults to "clouds".

            figsize (tuple, optional): Size of wordclouds. Defaults to (50, 30).

            facecolor (str, optional): Colour of wordcloud Defaults to "k".

            width (int, optional): width of plots. Defaults to 2000.

            height (int, optional): height of plots. Defaults to 1000.

            background_color (str, optional): Background colour of wordcloud.
                Defaults to "white".

            topic (int, optional): IF only one wordcloud is plotted, index of
                topic that is plotted. Defaults to 0.

            words (int, optional): Number of words per cloud. Defaults to 100.

            save (bool, optional): whether the plots should be saved or not.
                Defaults to False.

            n_clouds (int, optional): Number of word clouds that are plotted.
                Defaults to 1.

        Raises:
            ValueError: If save_name is not a string: no "Please specify a
                string as the name under which the plots should be saved"
        """

        if type == "pyldavis":
            # siehe: https://pyldavis.readthedocs.io/en/latest/modules/API.html
            # doc_lengths = list(
            #   preprocessed_column_target.apply(lambda x: len(x))
            #   )
            visualization = pyLDAvis.gensim_models.prepare(
                lda_model, bow_corpus, dictionary, sort_topics=False
            )
            pyLDAvis.save_html(visualization, "your_latest_lda_visualization.html")
            webbrowser.open("your_latest_lda_visualization.html")

        if type == "clouds" and n_clouds > len(lda_model.print_topics()):
            raise ValueError(
                "n_clouds must be <= the number of topics in your LDA computation"
            )

        if type == "clouds" and n_clouds <= 1:
            plt.figure(figsize=figsize, facecolor=facecolor)
            plt.imshow(
                WordCloud(
                    width=width, height=height, background_color=background_color
                ).fit_words(dict(lda_model.show_topic(topic, words)))
            )
            plt.axis("off")
            plt.title("Lda Model " + "topic #" + str(topic))
            plt.tight_layout(pad=0)

            plt.show()

        def _wordclouds(topic):
            wordcloud = WordCloud(
                width=width, height=height, background_color=background_color
            ).fit_words(dict(lda_model.show_topic(topic, words)))
            return wordcloud

        if type == "clouds" and n_clouds > 1:
            # initiliaze figure
            fig = plt.figure(figsize=figsize)
            # different plot sizes for different number of clouds
            if n_clouds % 2 != 0 and n_clouds >= 10:
                for i in range(n_clouds):
                    ax = fig.add_subplot(int(n_clouds / 3), int(n_clouds / 3), i + 1)
                    wordcloud = _wordclouds(i)
                    ax.imshow(wordcloud)
                    ax.axis("off")

            elif n_clouds % 2 == 0 and n_clouds < 10:
                for i in range(n_clouds):
                    ax = fig.add_subplot(2, int(n_clouds / 2), i + 1)
                    wordcloud = _wordclouds(i)
                    ax.imshow(wordcloud)
                    ax.axis("off")

            else:
                for i in range(n_clouds):
                    ax = fig.add_subplot(3, int(n_clouds / 2), i + 1)
                    wordcloud = _wordclouds(i)
                    ax.imshow(wordcloud)
                    ax.axis("off")

            plt.show()

        if save:
            if not isinstance(save_name, str):
                raise ValueError(
                    "Please specify a string as the name under which the plots should be saved"
                )

            plt.savefig(save_name)
