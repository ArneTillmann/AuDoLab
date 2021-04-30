import pyLDAvis.gensim_models
import pyLDAvis
from gensim import corpora, models


class LDA:
    """
    performs Latent Dirichlet Allocation Topic Modelling
    """

    def __init__(self):
        3 + 4

    @staticmethod
    def _preperation(df_processed, no_below=None, no_above=None):
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

        dictionary = corpora.Dictionary(df_processed["tokens"])
        if no_below is None:
            no_below = 0
        if no_above is None:
            no_above = 1
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        bow_corpus = [dictionary.doc2bow(doc) for doc
                      in df_processed["tokens"]]
        return dictionary, bow_corpus

    @staticmethod
    def model(corpus, num_topics, id2word, random_state, passes):
        """LDA model

        :param corpus: Stream of document vectors or sparse matrix of shape
        :type corpus: iterable of list of (int, float), optional
        :param num_topics: pre-defined number of topics
        :type num_topics: int
        :param id2word: Mapping from word IDs to words. It is
            used to determine the vocabulary size, as well as
            for debugging and topic printing.
        :type id2word: dict of (int, str): gensim.corpora.dictionary.Dictionary
        :param random_state: for recreating exact identical output
        :type random_state: int
        :param passes: Number of passes through the corpus during training.
        :type passes: int

        :return: returns lda_model output
        :rtype: lda_model
        """

        lda_model = models.LdaMulticore(
            corpus,
            num_topics=num_topics,
            id2word=id2word,
            random_state=random_state,
            passes=passes,
        )

        return lda_model

    @staticmethod
    def visualize_topics(lda_model, bow_corpus, dictionary):
        """Create pyLDAvis plots for LDA model output


        :param lda_model: Output of lda_model
        :type lda_model: lda_model
        :param bow_corpus: BoW representation of document.
        :type bow_corpus: list of (int, int)
        :param dictionary: Dict used for creating Corpus
        :type dictionary: gensim.corpora.dictionary.Dictionary

        :return: Html code of the visualization
        :rtype: IPython.Html object
        """

        visualization = pyLDAvis.gensim_models.prepare(
            lda_model, bow_corpus, dictionary, sort_topics=False
        )

        return pyLDAvis.display(visualization)
