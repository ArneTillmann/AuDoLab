import pyLDAvis.gensim_models
import pyLDAvis
from gensim import corpora, models
from pprint import pprint
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np


class LDA:
    """
    performs Latent Dirichlet Allocation Topic Modelling
    """

    def __init__(self):
        3 + 4

    @staticmethod
    def preperation(df_processed, no_below=None, no_above=None, column="preprocessed"):
        """Preprocessing for LDA

        Args:
            df_processed ([DataFrame]): [DataFrame stat store tokenized text in
                                        col ["tokens"]]
            no_below ([int]): [if word appears in less than
                              no_below documents they are not considered]
            no_above ([float]): [if e.g. 0.9, no words are taken into
                                account that appear more often than 90%]

        Returns:
            [dictionary]: [returns the dictionary used for LDA]
            [bow_corpus]: [Corpus used for LDA]
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
        corpus (iterable of list of (int, float), optional):
        [Stream of document vectors or sparse matrix of shape]
        num_topics (int): [pre-defined number of topics]
        id2word ({dict of (int, str): gensim.corpora.dictionary.Dictionary})
                                         –Mapping from word IDs to words. It is
                                        used to determine the vocabulary size,
                                as well as for debugging and topic printing.
        random_state (int): [for recreating exact identical output]
        passes (int): [Number of passes through the corpus during training.]

        Returns:
            [lda_model]: [returns lda_model output]
        """

        if multi == True:
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

        pprint(lda_model.top_topics(corpus))

        return lda_model

    @staticmethod
    def visualize_topics(
        lda_model,
        bow_corpus,
        dictionary,
        save_name="audolab_model.png",
        type="pyldavis",
        figsize=(50, 30),
        facecolor="k",
        width=2000,
        height=1000,
        background_color="white",
        topic=0,
        words=100,
        save=False,
        n_clouds=1,
    ):
        """Create pyLDAvis plots for LDA model output

        Args:
            lda_model ([type]): Output of lda_model
            bow_corpus ([list of (int, int)]): BoW representation of document.
            dictionary (gensim.corpora.dictionary.Dictionary): [Dict used for
                                                               creating Corpus]
        """

        if type == "pyldavis":
            visualization = pyLDAvis.gensim_models.prepare(
                lda_model, bow_corpus, dictionary, sort_topics=False
            )

            pyLDAvis.show(visualization)

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

        def _display_wordclouds(n_components):
            fig = plt.figure(figsize=(2, 2), facecolor="w", edgecolor="k")

            for t in range(n_components):
                temp = 251 + t  # this is to index the position of the subplot
                ax = plt.subplot(temp)

                ax.imshow(
                    WordCloud(
                        width=width, height=height, background_color=background_color
                    ).fit_words(dict(lda_model.show_topic(t, words)))
                )
                plt.axis("off")
            plt.show()

        def _wordclouds(topic):
            wordcloud = WordCloud(
                width=width, height=height, background_color=background_color
            ).fit_words(dict(lda_model.show_topic(topic, words)))
            return wordcloud

        if type == "clouds" and n_clouds >= 1:
            fig = plt.figure()
            if n_clouds >= 10:
                for i in range(n_clouds):
                    ax = fig.add_subplot(int(n_clouds / 3), int(n_clouds / 3), i + 1)
                    wordcloud = _wordclouds(i)
                    ax.imshow(wordcloud)
                    ax.axis("off")
            else:
                for i in range(n_clouds):
                    ax = fig.add_subplot(2, int(n_clouds / 2), i + 1)
                    wordcloud = _wordclouds(i)
                    ax.imshow(wordcloud)
                    ax.axis("off")

            plt.show()

        if save:
            if type(save_name) != str:
                raise ValueError(
                    "Please specify a string as the name under which the plots needs to be saved"
                )

            plt.savefig(save_name)
