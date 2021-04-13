import re
import webbot
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


sub = " Abstract:\n"


class AbstractScraper:
    """
    Abstractscraper - scrapes through IEEE Xplore and, depending on the given
    search query, scrapes the abstracts of scientific papers
    """

    def __init__(self, url):
        # define url - is already the result of search for the keyword
        # "Artificial Intelligence"
        self.url = url
        self.web = webbot.Browser()
        # define a delay and wait function in order to prevent empty results
        # when internet connection is slow
        self.delay = 3
        self.wait = 6

    def open(self):
        """goes through the number of defined pages on IEEE Xplore"""
        # go to web page
        self.web.go_to(self.url)

        time.sleep(self.delay)
        self.html_page = []
        self.html_page.append(self.web.get_page_source())

        time.sleep(self.wait)

        for i in range(1, 3):
            time.sleep(self.wait)
            self.web.click(str(i))
            time.sleep(self.wait)
            self.html_page.append(self.web.get_page_source())

        print("The algorithm is iterating through", len(self.html_page),
              "pages")
        return self.html_page

    def find_links(self):
        """goes through every paper on every page and collects all links to the
         subpages of the papers"""
        document_links = []
        self.data = []
        for j in range(len(self.html_page)):
            BeautifulSoup(self.html_page[j], features="html.parser")
            time.sleep(self.wait)

            # get the hyperlinks for all the documents and temporarily save
            # them
            for link in BeautifulSoup(
                self.html_page[j], features="html.parser"
            ).findAll("a", attrs={"href": re.compile("^/document")}):
                document_links.append(link.get("href"))
            time.sleep(self.wait)

        # remove unnecessary results of the href search
        matching = [s for s in document_links if "citation" in s]
        x = [i for i in document_links if i not in matching]
        self.data = x
        time.sleep(self.wait)

        self.data = np.array(self.data)
        # remove duplicates that are in there due to multiple occurrence in the
        # href
        self.data = np.unique(self.data)
        print("Total number of abstracts that will be scraped:",
              len(self.data))
        return self.data

    def get_abstracts(self):
        """Opens all links for the webpages for each paper and scrapes
         the paper's abstract"""

        self.abstracts = []

        # go through every search result and do the following: open the
        # keywords section,
        # extract the keywords (+ unnecessary stuff) ,append the keywords to
        # self.keys
        for i in range(len(self.data)):

            self.web.go_to("https://ieeexplore.ieee.org" + self.data[i])
            time.sleep(self.delay)

            html_page = self.web.get_page_source()
            soup = BeautifulSoup(html_page, features="html.parser")
            time.sleep(self.delay)

            texts = soup.find_all("div", {"class": "u-mb-1"})
            for t in texts:
                text = t.text.strip()
                text = text.replace("Abstract:\n", " ")
                df_dict = {"text": text}
            self.abstracts.append(df_dict)

        self.abstracts = pd.DataFrame(self.abstracts)
        self.paper_cleaning()
        return self.abstracts

    def paper_cleaning(self):
        self.abstracts = self.abstracts.drop_duplicates(subset=["text"])
        mistake = self.abstracts["text"].iloc[77]
        self.abstracts = self.abstracts[self.abstracts["text"] != mistake]


if __name__ == "__main__":

    # Execute the above code
    ks = AbstractScraper(
        "https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22:cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1"
    )

    html_code = ks.open()
    links = ks.find_links()
    abstracts = ks.get_abstracts()

    abstracts.to_csv("cotton.txt", header=True, index=False)
