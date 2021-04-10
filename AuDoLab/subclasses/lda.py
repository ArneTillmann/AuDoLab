import pyLDAvis.gensim
import pyLDAvis
from gensim import corpora, models


class LDA:
    """
    performs Latent Dirichlet Allocation Topic Modelling
    """

    def __init__(self):
        3 + 4

    @staticmethod
    def preperation(df_processed, no_below=None, no_above=None):
        """Preprocessing for LDA

        Args:
            df_processed ([DataFrame]): [DataFrame stat store tokenized text in col ["tokens"]]
            no_below ([int]): [if word appears in less than no_below documents they are not considered]
            no_above ([float]): [if e.g. 0.9, no words are taken into account that appear more often than 90%]

        Returns:
            [dictionary]: [returns the dictionary used for LDA]
            [bow_corpus]: [Corpus used for LDA]
        """

        dictionary = corpora.Dictionary(df_processed["tokens"])
        if no_below == None:
            no_below = 0
        if no_above == None:
            no_above = 1
        bow_corpus = [dictionary.doc2bow(doc) for doc in df_processed["tokens"]]
        return dictionary, bow_corpus

    #####---- Should not be necessary anymore ----############



    @staticmethod
    def model(corpus, num_topics, id2word, random_state, passes):
        """LDA model

        Args:
            corpus (iterable of list of (int, float), optional): [Stream of document vectors or sparse matrix of shape]
            num_topics (int): [pre-defined number of topics]
            id2word ({dict of (int, str): gensim.corpora.dictionary.Dictionary}) –
                                            Mapping from word IDs to words. It is
                                            used to determine the vocabulary size,
                                            as well as for debugging and topic printing.
            random_state (int): [for recreating exact identical output]
            passes (int): [Number of passes through the corpus during training.]

        Returns:
            [lda_model]: [returns lda_model output]
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

        Args:
            lda_model ([type]): Output of lda_model
            bow_corpus ([list of (int, int)]): BoW representation of document.
            dictionary (gensim.corpora.dictionary.Dictionary): [Dict used for creating Corpus]
        """

        visualization = pyLDAvis.gensim.prepare(
            lda_model, bow_corpus, dictionary, sort_topics=False
        )

        pyLDAvis.show(visualization)
