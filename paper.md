---
title: 'AuDoLab: Automatic document labelling and classfication for extremely unbalanced data'
tags:
  - Python
  - One-class SVM
  - Unsupervised Document Classification
  - One-class Document Classification
  - LDA Topic Modelling
  - Out-of-domain Training Data
authors:
  - name: Arne Tillmann^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Anton Thielmann
    affiliation: 1
  - name: Christoph Weisser
    affiliation: 1,2
  - name: Benjamin Säfken
    affiliation: 1,2
  - name: Thomas Kneib
    affiliation: 1,2
  - name: Alexander Silbersdorff
    affiliation: 1
affiliations:
 - name: Georg-August-Universität Göttingen, Göttingen, Germany
   index: 1
 - name: Campus-Institut Data Science (CIDAS), Göttingen, Germany
   index: 2
date: 24 April 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

AuDoLab provides a novel approach to one-class document classification for heavily imbalanced datasets, even if labelled training data is not available. 
AuDoLab enables the user to create user specific out-of-domain training data and classify a heavily underrepresented target class
in a target dataset using a recently developed integration of Web Scraping, Latent Dirichlet Allocation Topic Modelling and one-class Support Vector Machines [@Thielmann]. The user can achieve high quality results even on higly specific classification problems without the need to invest in the time and cost intensive 
labelling of training documents by humans. Hence, AuDoLab has a broad range of scientific research or business application.


The following section provides an detailed overview of AuDoLab. Subsequently, it will be discussed how the
theoretical models behind AuDoLab advance existing methods and software solutions. AuDoLab can be installed conveniently via pip.


installation and the package can be found in the packages repository or on the documentation website of TTLocVis test


# Statement of need

Unsupervised document classification is mainly performed to gain insight into the underlying topics of large text corpora.
In this process, highly underrepresented topics are often overlooked and consequently assigned to the wrong topics.
Thus, labeling underrepresented topics in large text corpora is often done manually and can therefore be very time-consuming.
AuDoLab enables the user to tackle this problem and perform unsupervised one-class document classification for heavily underrepresented document classes.
This leverages the results of one-class document classification using one-class support vector machines (SVM) [@Scholkopf][@Manevitz] and extends them to the use case of severely imbalanced datasets.
This adaptation and extension is achieved by implementing a multi-level classification rule as visualised in the graph below.

![Classification Procedure.\label{fig:test2}](figures/tree.PNG){ width=100% }


Firstly, the package enables the user to web scrape training documents (scientific papers) from IEEEXplore. The user can search for multiple search terms and specify an individual search query. Thus, the user can create its own, individually labelled (e.g. via author-keywords) training data set. Through the integration of pre-labelled out-of-domain training data, the problem of the heavily underrepresented target class can be circumvented, as large enough training corpora can be automatically generated.
Subsequently, the text data is preprocessed for the classification part. The text preprocessing includes common NLP text preprocessing techniques such as stopword removal and lemmatization.  As  document  representations  the  term  frequency-inverse  document  frequency  (tf-idf) representations are chosen. The tf-idf scores are computed on a joint corpus from the web-scraped out-of-domain training data and the target text data.

The main part of the classification rule lies in the training of the one-class SVM [@Scholkopf]. As a training corpus, only the out-of-domain training data is used.  By adjusting hyperparameters, the user can create a strict or relaxed classification rule, that reflects the users belief about the prevalence of the target class inside the target data set and the quality of the scraped out-of-domain training data. The last part of the classification rule enables the user to control the classifiers results with the help of LDA topic models [@Blei] (and e.g. wordclouds). Additionally, the user can generate interactive plots depicicting the identified topics during the LDA topic modelling [@ldavis].

The second step can be reiteraded, depending on the users perceived quality of the classification results.

# Comparison with existing tools





# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
