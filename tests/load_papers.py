import pandas as pd


papers = pd.read_csv(r"tests\dentistry_teeth.txt")
papers.head()
papers = papers.drop_duplicates(subset=["text"])
mistake = papers["text"].iloc[77]
papers = papers[papers["text"] != mistake]
if __name__ == "__main__":
    print(papers)
