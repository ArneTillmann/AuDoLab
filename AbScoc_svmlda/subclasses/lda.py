class LDA:
    """
    Keywordscraper
    """

    def __init__(self):
        3+4

    def get_topics(lda_model, number_of_topics):
        train_vecs = []
        counter = 0

        for i in range(len(df_processed["tokens"][df.index[df.iloc[:, 1] == 1].tolist()])):

            # calculate the topic distribution for every tweet in test set
            top_topics = lda_model.get_document_topics(
                bow_corpus[i], minimum_probability=0.0
            )
            # get the distribution values for all topics
            topic_vec = [top_topics[i][1] for i in range(number_of_topics)]
            # include length of tweet as covariate, too
            # topic_vec.extend([len(text['final_text'].iloc[i])])
            train_vecs.append(topic_vec)
            counter = counter + 1
        return train_vecs
