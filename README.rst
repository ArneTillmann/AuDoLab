=======
AuDoLab
=======

.. image:: https://img.shields.io/pypi/v/AuDoLab.svg
        :target: https://pypi.python.org/pypi/AuDoLab

.. image:: https://api.travis-ci.com/ArneTillmann/AuDoLab.svg?branch=main&status=passed
        :target: https://travis-ci.com/ArneTillmann/AuDoLab

.. image:: https://readthedocs.org/projects/audolab/badge/?version=latest
 :target: https://audolab.readthedocs.io/en/latest/?badge=latest
 :alt: Documentation Status

With AuDoLab you can perform Latend Direchlet Allocation on highly imbalanced datasets.

============
Installation
============


Stable release
--------------

To install AuDoLab, run this command in your terminal:

.. code-block:: console

    $ pip install AuDoLab

This is the preferred method to install AuDoLab, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for AuDoLab can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ArneTillmann/AuDoLab

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/ArneTillmann/AuDoLab/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/ArneTillmann/AuDoLab
.. _tarball: https://github.com/ArneTillmann/AuDoLab/tarball/master

=====
Usage
=====
Before the actuall usage you want to download the stopwords for nltk by running::

    import nltk
    nltk.download('stopwords')

inside a python console.
To use AuDoLab in a project::

    from AuDoLab import AuDoLab
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()
    from numpy import round as np_round
    from numpy import arange as np_arange

Then you want to create an instance of the AuDoLab class

    audo = AuDoLab.AuDoLab()

In this example we used publicly available data from the nltk package::

    from nltk.corpus import reuters
    import numpy as np
    import pandas as pd

    data = []

    for fileid in reuters.fileids():
        tag, filename = fileid.split("/")
        data.append(
            (filename,
             ", ".join(
                 reuters.categories(fileid)),
                reuters.raw(fileid)))

    data = pd.DataFrame(data, columns=["filename", "categories", "text"])

Then you want to scrape abstracts, e.g. from IEEE with the abstract scraper::

    async def scrape():
        return await audo.scrape_abstracts(
            url=None, keywords=["cotton"], in_data="all_meta", pages=5
        )

    scraped_documents = asyncio.get_event_loop().run_until_complete(scrape())

The data as well as the scraped papers need to be preprocessed before use in the
classifier::

    preprocessed_target = audo.preprocessing(data=data, column="text")

    preprocessed_paper = audo.preprocessing(
        data=scraped_documents, column="text")

    target_tfidf, training_tfidf = audo.tf_idf(
        data=preprocessed_target,
        papers=preprocessed_paper,
        data_column="lemma",
        papers_column="lemma",
        features=100000,
    )

Afterwards we can train and use the classifiers and choose the desired
one::

    classifier = audo.one_class_svm(
        training=training_tfidf,
        predicting=target_tfidf,
        nus=np.round(np.arange(0.01, 0.5, 0.01), 7),
        quality_train=0.9,
        min_pred=0.001,
        max_pred=0.05,
    )

    df_data = audo.choose_classifier(preprocessed_target, classifier, 2)

And finally you can estimate the topics of the data::

    audo.lda_modeling(df_data, num_topics=2)

    a = audo.lda_visualize_topics()
    html = a.data
    with open('html_file.html', 'w') as f:
        f.write(html)

* Free software: GNU General Public License v3
* Documentation: https://AuDoLab.readthedocs.io.
