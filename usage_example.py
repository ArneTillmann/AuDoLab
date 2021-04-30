from AuDoLab import AuDoLab
import asyncio
from numpy import round as np_round
from numpy import arange as np_arange


audo = AuDoLab.AuDoLab()

if __name__ == "__main__":

    # Load target data
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

    data = pd.DataFrame(data, columns=["filename", "categories", "text"])


####### ----------------------------------------------------------------------------- #######
    preprocessed_target = audo.preprocessing(data=data, column="text")

    async def scrape():
        return await audo.scrape_abstracts(
            url=None, keywords=["cotton"], in_data="all_meta", pages=5
        )

    scraped_documents = asyncio.get_event_loop().run_until_complete(scrape())

    preprocessed_paper = audo.preprocessing(
        data=scraped_documents, column="text")

    target_tfidf, training_tfidf = audo.tf_idf(
        data=preprocessed_target,
        papers=preprocessed_paper,
        data_column="lemma",
        papers_column="lemma",
        features=100000,
    )

    classifier = audo.one_class_svm(
        training=training_tfidf,
        predicting=target_tfidf,
        nus=np.round(np.arange(0.01, 0.5, 0.01), 7),
        quality_train=0.9,
        min_pred=0.001,
        max_pred=0.05,
    )

    df_data = audo.choose_classifier(preprocessed_target, classifier, 2)

    audo.lda_modeling(df_data, num_topics=2)
    
    a = audo.lda_visualize_topics()
    html = a.data
    with open('html_file.html', 'w') as f:
        f.write(html)
