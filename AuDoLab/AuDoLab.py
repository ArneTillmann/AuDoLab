from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf

class AuDoLab:

    def __init__(self):
        4+5

    def scrape_abstracts(self, url):
        ks = AbstractScraper(url)

        html_code = ks.open()
        links = ks.find_links()
        self.abstracts = ks.get_abstracts()

        self.abstracts.to_csv("cotton.txt", header=True, index=False)

    def tf_idf_features(self, data, papers=self.abstracts):
        tf_idf = Tf_idf()
        self.data_tfidf_features, self.papers_tfidf_features = tf_idf.tfidf_features(
            data, papers)

    def tf_idf(self, data, papers=self.abstracts):

        tf_idf = Tf_idf()
        self.data_tfidf, self.papers_tfidf = tf_idf.tfidf(data, papers)

    def one_class_svm(self, nus=np.round(np.arange(0.001, 0.5, 0.001), 5), quality_train=0.85, min_pred=0.01, max_pred=0.1):
        self.one_Class_SVM = One_Class_SVM()
        classifier = one_Class_SVM.classification(
            training=papers_tfidf,
            predicting=data_tfidf,
            nus=nus,
            quality_train=quality_train,
            min_pred=min_pred,
            max_pred=max_pred,
        )

    def choose_classifier(self, i, df=self.one_Class_SVM):
        return One_Class_SVM.choose_classifier(df, i)

    def lda_modeling(self):
        dictionary_paper, bow_corpus_paper = LDA.preperation(papers_processed, no_below=2, no_above=0.9)
        self.lda = LDA()
        lda_model_paper = lda.model(
            bow_corpus_paper,
            num_topics=3,
            id2word=dictionary_paper,
            random_state=101,
            passes=10,
            )
