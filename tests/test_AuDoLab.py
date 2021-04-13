import pytest
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from AuDoLab import AuDoLab
"""Tests for `AuDoLab` package."""


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


audo = AuDoLab.AuDoLab()
# # papers = audo.scrape_abstracts(
# #     keywords=["dentistry", "teeth", "tooth"],
# #     in_data="all_meta",
# #     pages=12,
# #     operator="or"
# # )
# papers_processed = audo.preprocessing(papers, "text")
# data_processed = audo.preprocessing(data, "transcription")
#
# data_tfidf, papers_tfidf = audo.tf_idf(
#     data, papers,"transcription", "text")
# classifier = audo.one_class_svm(papers_tfidf, data_tfidf)
# df_data = audo.choose_classifier(data_processed, classifier, 0)
# if __name__ == '__main__':
#     # lda = audo.lda_modeling(papers_processed)
#     # audo.lda_visualize_topics()
#     lda2 = audo.lda_modeling(df_data, no_above=0.3)
#     audo.lda_visualize_topics()
