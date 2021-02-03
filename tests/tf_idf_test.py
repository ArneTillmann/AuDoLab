import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from AbScoc_svmlda.subclasses.tf_idf import Tf_idf
from load_papers import papers
from load_data import data


tf_idf = Tf_idf()
data_tfidf_features, papers_tfidf_features = tf_idf.tfidf_features(
    data, papers)
data_tfidf, papers_tfidf = tf_idf.tfidf(data, papers)
if __name__ == "__main__":
    print(data_tfidf.shape, data_tfidf_features.shape)
