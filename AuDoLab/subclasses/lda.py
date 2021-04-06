import gensim
import pyLDAvis.gensim
import pyLDAvis


class LDA:
    """
    Keywordscraper
    """

    def __init__(self):
        3+4

    # def get_topics(lda_model, number_of_topics):
    #     train_vecs = []
    #     counter = 0
    #
    #     for i in range(len(df_processed["tokens"][df.index[df.iloc[:, 1] == 1].tolist()])):
    #
    #         # calculate the topic distribution for every tweet in test set
    #         top_topics = lda_model.get_document_topics(
    #             bow_corpus[i], minimum_probability=0.0
    #         )
    #         # get the distribution values for all topics
    #         topic_vec = [top_topics[i][1] for i in range(number_of_topics)]
    #         # include length of tweet as covariate, too
    #         # topic_vec.extend([len(text['final_text'].iloc[i])])
    #         train_vecs.append(topic_vec)
    #         counter = counter + 1
    #     return train_vecs

    @staticmethod
    def preperation(df_processed, no_below, no_above):

        dictionary = gensim.corpora.Dictionary(df_processed["tokens"])
        dictionary.filter_extremes(no_above=no_above, no_below=no_below)
        bow_corpus = [dictionary.doc2bow(doc)
                      for doc in df_processed["tokens"]]
        return dictionary, bow_corpus

    @staticmethod
    def preperation2(df_processed, no_above):
        dictionary = gensim.corpora.Dictionary(df_processed["tokens"])
        dictionary.filter_extremes(no_above=no_above)
        bow_corpus = [dictionary.doc2bow(doc)
                      for doc in df_processed["tokens"]]
        return dictionary, bow_corpus

    @staticmethod
    def preperation(df_processed, no_below):
        dictionary = gensim.corpora.Dictionary(df_processed["tokens"])
        dictionary.filter_extremes(no_below=no_below)
        bow_corpus = [dictionary.doc2bow(doc)
                      for doc in df_processed["tokens"]]
        return dictionary, bow_corpus

    @staticmethod
    def preperation(df_processed):
        dictionary = gensim.corpora.Dictionary(df_processed["tokens"])
        bow_corpus = [dictionary.doc2bow(doc)
                      for doc in df_processed["tokens"]]
        return dictionary, bow_corpus

    @staticmethod
    def model(corpus, num_topics, id2word,
              random_state, passes):

        lda_model = gensim.models.LdaMulticore(
            corpus,
            num_topics=num_topics,
            id2word=id2word,
            random_state=random_state,
            passes=passes,
        )

        return lda_model

    @staticmethod
    def visualize_topics(lda_model, bow_corpus, dictionary):

        visualization = pyLDAvis.gensim.prepare(
            lda_model, bow_corpus, dictionary, sort_topics=False
        )

        pyLDAvis.show(visualization)
