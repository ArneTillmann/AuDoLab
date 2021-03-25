import gensim
from preprocessing_test import papers_processed
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from AuDoLab.subclasses.lda import LDA
# from multiprocessing import freeze_support, main

dictionary_paper, bow_corpus_paper = LDA.preperation(papers_processed, no_below=2, no_above=0.9)

if __name__ == '__main__':
    lda = LDA()
    lda_model_paper = lda.model(
        bow_corpus_paper,
        num_topics=3,
        id2word=dictionary_paper,
        random_state=101,
        passes=10,
        )
    for idx, topic in lda_model_paper.print_topics(-1):
        print("Topic: {} Word: {}".format(idx, topic))
    LDA.visualize_topics(lda_model_paper, bow_corpus_paper, dictionary_paper)
