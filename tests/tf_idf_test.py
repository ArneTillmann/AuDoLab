from load_data import data
from load_papers import papers
from AuDoLab.subclasses.tf_idf import Tf_idf
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")


tf_idf = Tf_idf()
data_tfidf_features, papers_tfidf_features = tf_idf.tfidf_features(
    data, papers)
data_tfidf, papers_tfidf = tf_idf.tfidf(data, papers)
if __name__ == "__main__":
    print(data_tfidf.shape, data_tfidf_features.shape)
