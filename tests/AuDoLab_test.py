import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from AuDoLab.AuDoLab import AuDoLab

from load_papers import papers
from load_data import data

audo = AuDoLab()
#papers2 = audo.scrape_abstracts("https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22:cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1")
papers_processed = audo.preprocessing(papers)
data_processed = audo.preprocessing(data)
data_tfidf_features, papers_tfidf_features = audo.tf_idf_features(data, papers)
data_tfidf, papers_tfidf = audo.tf_idf(data, papers)
classifier = audo.one_class_svm(papers_tfidf, data_tfidf)
df_data = audo.choose_classifier(data_processed, classifier, 0)
if __name__ == '__main__':
    # lda = audo.lda_modeling(papers_processed)
    # audo.lda_visualize_topics()

    lda2 = audo.lda_modeling(df_data, no_above=0.3)
    audo.lda_visualize_topics()
