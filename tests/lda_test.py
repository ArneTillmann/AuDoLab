import gensim
from preprocessing_test import papers_processed
from multiprocessing import freeze_support, main

dictionary_paper = gensim.corpora.Dictionary(papers_processed["tokens"])
dictionary_paper.filter_extremes(no_below=2, no_above=0.9)
bow_corpus_paper = [dictionary_paper.doc2bow(doc) for doc in papers_processed["tokens"]]

if __name__ == '__main__':
    main()
    lda_model_paper = gensim.models.LdaMulticore(
        bow_corpus_paper,
        num_topics=3,
        id2word=dictionary_paper,
        random_state=101,
        passes=10,
        )

# for idx, topic in lda_model_paper.print_topics(-1):
#     print("Topic: {} Word: {}".format(idx, topic))
