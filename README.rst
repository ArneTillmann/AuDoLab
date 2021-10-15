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

=======
Summary
=======

AuDoLab provides a novel approach to one-class document classification for heavily imbalanced datasets, even if labelled training data is not available. Our package enables the user to create specific out-of-domain training data to classify a heavily underrepresented target class in a document dataset using a recently developed integration of Web Scraping, Latent Dirichlet Allocation Topic Modelling and One-class Support Vector Machines. AuDoLab can achieve high quality results even on higly specific classification problems without the need to invest in the time and cost intensive labelling of training documents by humans. Hence, AuDoLab has a broad range of scientific research or business applications.

Unsupervised document classification is mainly performed to gain insight into the underlying topics of large text corpora. In this process, documents covering highly underrepresented topics have only a minor impact on the algorithm's topic definitions. As a result, underrepresented topics can sometimes be "overlooked" and documents are assigned topic prevalences that do not reflect the underlying content. Thus, labeling underrepresented topics in large text corpora is often done manually and can therefore be very labour-intensive and time-consuming. AuDoLab enables the user to tackle this problem and perform unsupervised one-class document classification for heavily underrepresented document classes.

============
Installation
============


Stable release
--------------

To install AuDoLab, run this command in your terminal (bash, PowerShell, etc.), given that you have python 3 and pip installed :

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

    scraped_documents = audo.get_ieee("https://ieeexplore.ieee.org/search
                                       /searchresult.jsp?newsearch=true&
                                       queryText=cotton&highlight=true&
                                       returnFacets=ALL&returnType=SEARCH&
                                       matchPubs=true&rowsPerPage=100&
                                       pageNumber=1\",
                                       pages=1)

The data as well as the scraped papers need to be preprocessed before use in the
classifier::

    preprocessed_target = audo.text_cleaning(data=data, column="text")

    preprocessed_paper = audo.text_cleaning(
        data=scraped_documents, column="abstract")

    target_tfidf, training_tfidf = audo.tf_idf(
        data=preprocessed_target,
        papers=preprocessed_paper,
        data_column="lemma",
        papers_column="lemma",
        features=100000,
    )

Afterwards we can train and use the classifiers and choose the desired
one::

    o_svm_result = audo.one_class_svm(
        training=training_tfidf,
        predicting=target_tfidf,
        nus=np.round(np.arange(0.001, 0.5, 0.01), 7),
        quality_train=0.9,
        min_pred=0.001,
        max_pred=0.05,
    )

    result = audo.choose_classifier(preprocessed_target, o_svm_result, 0)

And finally you can estimate the topics of the data::

    lda_target = audo.lda_modeling(data=result, num_topics=5)

    audo.lda_visualize_topics(type="pyldavis")

* Free software: GNU General Public License v3
* Documentation: https://AuDoLab.readthedocs.io.
