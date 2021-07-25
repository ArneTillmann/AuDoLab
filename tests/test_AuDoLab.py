import pytest
# import os
# import sys
# from numpy import round as np_round
# from numpy import arange as np_arange
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
# from AuDoLab import AuDoLab
# """Tests for `AuDoLab` package."""


@pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     import requests
#     return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


# audo = AuDoLab.AuDoLab()
# # papers = audo.scrape_abstracts(
# #     keywords=["dentistry", "teeth", "tooth"],
# #     in_data="all_meta",
# #     pages=12,
# #     operator="or"
# # )
# papers_processed = audo.preprocessing(papers, "text")
# data_processed = audo.preprocessing(data, "transcription")
#
# data_tfidf, papers_tfidf = audo.tf_idf( data_processed, papers_processed, data_column="lemma", papers_column="lemma")

# nus = np.arange(0.001, 0.5, 0.001)
# nus = np.round(nus, 5)
#
# classifier = audo.one_class_svm(
#     training=papers_tfidf,
#     predicting=data_tfidf,
#     nus=[0.166,0.3],
#     quality_train=0.85,
#     min_pred=0.01,
#     max_pred=0.1,
# )
#
# df_data = audo.choose_classifier(data_processed, classifier, 0)
#
# if __name__ == '__main__':
#     # lda = audo.lda_modeling(papers_processed, num_topics=3,
#     # random_state=101,
#     # passes=10,
#     # no_below=2, no_above=0.9
#     # )
#
#     lda2 = audo.lda_modeling(df_data, no_above=0.3)
#
#     a = audo.lda_visualize_topics()
#     html = a.data
#     with open('html_file.html', 'w') as f:
#         f.write(html)
