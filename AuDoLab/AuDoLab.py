import numpy as np

from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf

class AuDoLab:

    def __init__(self):
        4+5

    def scrape_abstracts(self, url):
        ks = abstractscraper.AbstractScraper(url)

        html_code = ks.open()
        links = ks.find_links()
        self.abstracts = ks.get_abstracts()

        self.abstracts.to_csv("cotton.txt", header=True, index=False)
        return self.abstracts

    def preprocessing(self, data):

        self.data_processed = preprocessing.Preprocessor.basic_preprocessing(data)
        return self.data_processed

    def tf_idf_features(self, data, papers):
        tfidf = tf_idf.Tf_idf()
        self.data_tfidf_features, self.papers_tfidf_features = tfidf.tfidf_features(
            data, papers)
        return self.data_tfidf_features, self.papers_tfidf_features

    def tf_idf(self, data, papers):
        tfidf = tf_idf.Tf_idf()
        self.data_tfidf, self.papers_tfidf = tfidf.tfidf(data, papers)
        return self.data_tfidf, self.papers_tfidf

    def one_class_svm(self, training, predicting,
                      nus=np.round(np.arange(0.001, 0.5, 0.001), 5),
                      quality_train=0.85, min_pred=0.01, max_pred=0.1):
        one_Class_SVM = one_class_svm.One_Class_SVM()
        self.df = one_Class_SVM.classification(
            training=training,
            predicting=predicting,
            nus=nus,
            quality_train=quality_train,
            min_pred=min_pred,
            max_pred=max_pred,
        )
        return self.df

    def choose_classifier(self, df, classifier, i):
        return one_class_svm.One_Class_SVM.choose_classifier(df, classifier, i)


    def lda_preperation(df_processed, no_below, no_above):
        return lda.LDA.preperation(df_processed, no_below, no_above)

    def lda_modeling(self, data, num_topics=5,  random_state=101, passes=20):
        self.dictionary, self.bow_corpus = lda.LDA.preperation(data, no_below=2, no_above=0.3)
        self.l = lda.LDA()
        self.lda_model = self.l.model(
            self.bow_corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            random_state=random_state,
            passes=passes,
            )
        return self.lda_model

    def lda_visualize_topics(self):
        lda.LDA.visualize_topics(self.lda_model, self.bow_corpus, self.dictionary)
