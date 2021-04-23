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
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Anton Thielmann
    affiliation: 2
  - name: Christoph Weisser
    affiliation: 3
  - name: Benjamin Säfken
    affiliation: 3
  - name: Thomas Kneib
    affiliation: 3
  - name: Alexander Silbersdorff
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

In Natural Language Processing (NLP) Unsupervised Document Classification is mainly done on large and balanced datasets.
AuDoLab tackles this problem and provides a novel approach to one-class document classification for heavily imbalanced datasets.

Furthermore, AuDoLab enables the user to  create  user specific out-of-domain training data and classify a heavily underrepresented target class
in a target dataset, using Web Scraping, Latent Dirichlet Allocation Topic Modelling and one-class Support Vector Machines.




AuDoLab can be used for various applications for scientific research and also has various business applications.
The following section provides an detailed overview of AuDoLab. Subsequently, it will be discussed how the
theoretical models behind AuDoLab advance existing methods and software solutions. AuDoLab can be installed conveniently via pip.



 installation and the package can be found in the packages repository or on the documentation website of TTLocVis


# Statement of need

Unsupervised document classification is mainly performed to gain insight into the underlying topics of large text corpora.
In this process, highly underrepresented topics are often overlooked and consequently assigned to the wrong topics. Thus, labeling underrepresented topics in large text corpora is often done manually and can therefore be very time-consuming. 

![Classification Procedure.\label{fig:test2}](figures/tree.PNG){ width=100% }

AuDoLab enables the user to tackle this problem and perform unsupervised one-class document classification for heavily underrepresented document classes. 
Firstly, the package enables the user to web scrape training documents (scientific papers) from IEEEXplore. The user can search for multiple search terms and specify an individual search query. Subsequently, the text data is preprocessed for the classification part. The text preprocessing includes common NLP text preprocessing techniques as stopword removal and lemmatization.  As  document  representations  the  term  frequency-inverse  document  frequency  (tf-idf) representations are chosen. The tf-idf scores are computed on a joint corpus from the web-scraped out-of-domain training data and the target text data. 

The actual document classification is performed using one-class Support vector machines, trained on the tf-idf representations from the out-of-domain training data, that are computed on the joint corpus. 

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
