#!/usr/bin/env python

"""Tests for `AuDoLab` package."""

import pytest

from click.testing import CliRunner

import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from AuDoLab import cli
from AuDoLab import AuDoLab




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


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'AuDoLab.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


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
# data_tfidf, papers_tfidf = audo.tf_idf(
#     data, papers,"transcription", "text")
# classifier = audo.one_class_svm(papers_tfidf, data_tfidf)
# df_data = audo.choose_classifier(data_processed, classifier, 0)
# if __name__ == '__main__':
#     # lda = audo.lda_modeling(papers_processed)
#     # audo.lda_visualize_topics()
#     lda2 = audo.lda_modeling(df_data, no_above=0.3)
#     audo.lda_visualize_topics()
