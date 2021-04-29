from AuDoLab import AuDoLab
import pandas as pd
import asyncio
from numpy import round as np_round
from numpy import arange as np_arange


audo = AuDoLab.AuDoLab()

if __name__ == "__main__":

    # Load target data
    from nltk.corpus import reuters

    data = []

    for fileid in reuters.fileids():
        tag, filename = fileid.split("/")
        data.append(
            (filename,
             ", ".join(
                 reuters.categories(fileid)),
                reuters.raw(fileid)))

    data = pd.DataFrame(data, columns=["filename", "categories", "text"])

    preprocessed_target = audo.preprocessing(data=data, column="text")

    async def scrape():
        return await audo.scrape_abstracts(
            url=None, keywords=["cotton"], in_data="all_meta", pages=1)

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

    test = audo.one_class_svm(training=training_tfidf, predicting=target_tfidf, nus=np_round(np_arange(0.0001, 0.9, 0.0001), 7),
                            quality_train=0.01,
                            min_pred=0.001,
                            max_pred=0.99,)
