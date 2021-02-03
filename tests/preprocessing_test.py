import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from AbScoc_svmlda.subclasses.preprocessing import Preprocessor
from load_papers import papers
from load_data import data


preprocessing = Preprocessor()
papers_processed = Preprocessor.basic_preprocessing(papers)
if __name__ == "__main__":
    print(papers_processed)
df_processed = Preprocessor.basic_preprocessing(data)
if __name__ == "__main__":
    print(data)
