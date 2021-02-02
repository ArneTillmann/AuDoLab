import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from AbScoc_svmlda.subclasses.preprocessing import Preprocessor


preprocessing = Preprocessor()
papers = pd.read_csv(r"tests\dentistry_teeth.txt")
papers.head()
papers = papers.drop_duplicates(subset=["text"])
print(papers)
mistake = papers["text"].iloc[77]
papers = papers[papers["text"] != mistake]
papers_processed = Preprocessor.basic_preprocessing(papers)
print(papers_processed)
