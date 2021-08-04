from AuDoLab import AuDoLab

audo = AuDoLab.AuDoLab()

if __name__ == "__main__":

    # Load target data from reuters dataset
    from nltk.corpus import reuters
    import numpy as np
    import pandas as pd

    data = []

    for fileid in reuters.fileids():
        tag, filename = fileid.split("/")
        data.append(
            (filename,
             ", ".join(
                 reuters.categories(fileid)),
                reuters.raw(fileid)))

    # store loaded data in dataframe
    data = pd.DataFrame(data, columns=["filename", "categories", "text"])

    #####------
    # start using audolab

    # clean theloaded data
    preprocessed_target = audo.text_cleaning(data=data, column="text")

    # scrape ieee
    scraped_documents = audo.get_ieee(pages=1)

    # clean the scraped papers
    preprocessed_paper = audo.text_cleaning(data=scraped_documents, column="abstract")

    # calculate tfidf values on joint corpus
    target_tfidf, training_tfidf = audo.tf_idf(
        data=preprocessed_target,
        papers=preprocessed_paper,
        data_column="lemma",
        papers_column="lemma",
        features=100000,
    )

    # calculate one_class_svm on data
    o_svm_result = audo.one_class_svm(
        training=training_tfidf,
        predicting=target_tfidf,
        nus=np.round(np.arange(0.001, 0.5, 0.01), 7),
        quality_train=0.9,
        min_pred=0.001,
        max_pred=0.05,
    )

    # select a classifier
    result = audo.choose_classifier(preprocessed_target, o_svm_result, 0)

    # perform topic modeling and plot the created topics
    lda_target = audo.lda_modeling(data=result, num_topics=5)
    audo.lda_visualize_topics(type="clouds", n_clouds=4)
