import re
import webbot
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import json


class AbstractScraper:
    """
    Abstractscraper - scrapes through IEEE Xplore and, depending on the given
    search query, scrapes the abstracts of scientific papers
    """

    def __init__(self):
        self.web = webbot.Browser()
        self.wait = 5
        self.first = "https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&"
        self.second = "&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1"

    def _open(self, url=None, keywords=None, operator="OR", in_data="author",
              pages=2):
        """
        defines the users search query

        :param url: when the user specifies an own search query on IEEEXplore
        :type url: string
        :param keywords: keywords that are searched for
        :type keywords: list
        :param operator: "and" / "or" operator between keywords
        :type operator: string
        :param pages: number of pages that is iterated over
        :type pages: int
        :param in_data: "author" or "all_meta" whether to search in author
                        keywords or all metadata
        :type in_data: string
        :return:
        """
        operator = ")%20" + operator.upper() + "%20("
        if in_data == "all_meta":
            in_data = "%22All%20Metadata%22:"
        elif in_data == "author":
            in_data = "%22Author%20Keywords%22:"
        if url:
            self.url = url
            self.web.go_to(self.url)

        else:
            if len(keywords) > 3:
                print("You can only specify 3 Keywords/phrases")
            elif len(keywords) == 1:
                self.middle = "queryText=" + in_data + keywords[0]
            elif len(keywords) == 2:
                self.middle = (
                    "queryText=(("
                    + in_data
                    + keywords[0]
                    + operator
                    + in_data
                    + keywords[1]
                    + "))"
                )
            else:
                self.middle = (
                    "queryText=(("
                    + in_data
                    + keywords[0]
                    + operator
                    + in_data
                    + keywords[1]
                    + operator
                    + in_data
                    + keywords[2]
                    + "))"
                )
            self.url = self.first + self.middle + self.second
            self.web.go_to(self.url)
            time.sleep(self.wait)

        """goes through the number of defined pages on IEEE Xplore"""
        # go to web page
        self.web.click("Accept")
        self.html_page = []

        for i in range(1, pages + 1):
            self.web.click(str(i))
            time.sleep(self.wait)
            self.html_page.append(self.web.get_page_source())
            time.sleep(self.wait)
        print("The algorithm is iterating through", len(self.html_page),
              "pages")
        self.web.quit()

    def _find_links(self):
        """goes through every paper on every page and collects all links to the
         subpages of the papers"""
        document_links = []
        self.data = []
        for j in range(len(self.html_page)):
            for link in BeautifulSoup(
                self.html_page[j], features="html.parser"
            ).findAll("a", attrs={"href": re.compile("^/document")}):
                document_links.append(link.get("href"))

        # remove unnecessary results of the href search
        matching = [s for s in document_links if "citation" in s]
        x = [i for i in document_links if i not in matching]
        self.data = x
        self.data = np.array(self.data)
        # remove duplicates that are in there due to multiple occurrence in the
        self.data = np.unique(self.data)
        print("Total number of abstracts that will be scraped:",
              len(self.data))

    def _find_abstracts(self):
        """Opens all links for the webpages for each paper and scrapes the
        paper's abstract"""
        # initialize empty lists to fill
        self.abstracts = []
        self.title = []

        # loop through number of  "link"
        for i in range(len(self.data)):
            # only "try" because sometimes the javascript is corrupted
            try:
                data = json.loads(
                    re.search(
                        r"\.metadata=(.*?);",
                        requests.get("https://ieeexplore.ieee.org" +
                                     self.data[i]).text,
                    ).group(1)
                )
                # only get title and abstracts -> we could also go for
                # author etc.
                title = data["title"]
                data = data["abstract"]

                self.title.append(title)
                self.abstracts.append(data)
            except:
                pass

    def get_abstracts(
        self, url=None, keywords=None, operator="OR", pages=2, in_data="author"
    ):
        """
        :param url: when the user specifies an own search query on IEEEXplore
        :type url: string
        :param keywords: keywords that are searched for
        :type keywords: list
        :param operator: "and" / "or" operator between keywords
        :type operator: string
        :param pages: number of pages that is iterated over
        :type pages: int
        :param in_data: "author" or "all_meta" whether to search in author
                keywords or all metadata
        :type in_data: string
        :return: pd.DataFrame
        """
        self._open(
            url=url, keywords=keywords, operator=operator, pages=pages,
            in_data=in_data
        )
        self._find_links()
        self._find_abstracts()

        data = pd.DataFrame({"text": self.abstracts, "titles": self.title})
        data = data.drop_duplicates()
        return data


if __name__ == "__main__":
    # Execute the above code
    AS = AbstractScraper()
    data = AS.get_abstracts(
        keywords=["dentistry", "teeth", "tooth"],
        in_data="all_meta",
        pages=12,
        operator="or",
    )
    print(data["text"])
