import warnings
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from requests import get as requests_get
from json import loads as json_loads
import asyncio
from pyppeteer import launch
from tqdm import tqdm


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class AbstractScraper:
    """
    Abstractscraper - scrapes through IEEE Xplore and, depending on the given
    search query, scrapes the abstracts of scientific papers
    """

    def __init__(self):
        """Define url for when a search query in the form of keywords are given"""
        self.first = "https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&"
        self.second = "&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1"

    async def _open(
        self, url=None, keywords=None, operator="OR", in_data="author", pages=2
    ):
        """
        defines the users search query and pbuilds a suitable URL according to the specified keywords

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
        """

        await self._open_pypeteer(url, keywords, operator, in_data, pages)

    async def _open_pypeteer(
        self, url=None, keywords=None, operator="OR", in_data="author", pages=2
    ):
        """function that opens a web browser in the background and

        Args:
            url (string, optional): if given than the search defined in this url
                is scraped. Defaults to None.
            keywords (list, optional): f no url is given but a keyword list, a
                search after these keywords is performed and the results are
                scraped. Defaults to None.
            operator (str, optional): Operator between the keywords. If "AND"
                the search results must include all specified keywords. Defaults
                to "OR".
            in_data (str, optional): "author" or "all_meta" -> whether the
                keywords are only searched in the author keywords or in all
                metadata. Defaults to "author".
            pages (int, optional): number of pages that are iterated over.
                Defaults to 2.
        """

        self._create_url(url, keywords, operator, in_data)
        self.html_page = []

        # Print statement so everyone knows what is currently happening
        if pages == 1:
            print("The algorithm is iterating through", pages, "page")
        else:
            print("The algorithm is iterating through", pages, "pages")

        browser = await launch()
        page = await browser.newPage()
        await page.goto(self.url, {"waitUntil": "networkidle0"})

        self.linkstorage = []

        elements = await page.querySelectorAll("a")
        for element in elements:
            self.linkstorage.append(
                await page.evaluate("(element) => element.href", element)
            )
        for i in tqdm(range(2, pages + 1)):
            self._create_new_url(i)
            await page.goto(self.url, {"waitUntil": "networkidle0"})
            elements = await page.querySelectorAll("a")
            for element in elements:
                self.linkstorage.append(
                    await page.evaluate("(element) => element.href", element)
                )
        await browser.close()

    def _add_page_number(self, page, position):
        """helper func that extracts the pages from the users search query

        Args:
            page (int): helper variable for number of page
            position (int): helper variable for position at which it has already been scraped
        """
        self.url = (
            self.url[: position + len("pageNumber=")]
            + str(page)
            + self.url[position + len("pageNumber=") + len(str(page - 1)) :]
        )

    def _add_str_plus_page_number(self, page):
        """helper func to iterate through pages

        Args:
            page (int): page
        """
        if self.url[-1] == "?":
            self.url = self.url + "pageNumber=" + str(page)
        else:
            self.url = self.url + "?pageNumber" + str(page)

    def _create_new_url(self, page):
        """helper func

        Args:
            page (int): -
        """

        if "pageNumber=" in self.url:
            self._add_page_number(page, self.url.find("pageNumber="))
        else:
            self._add_str_plus_page_number(page)

    def _create_url(self, url, keywords, operator, in_data):
        """helper func to create the URL that is generated according to the useres keywords

        Args:
            url (str): if a url is given, the function is not building a new url
            keywords (list): if keywords are given a search query is implemented
            operator (str): operator between keywords, "AND" or "OR"
            in_data (str): either "author", or "all_meta - specifies whether the keywords must appear only in the author keywords or can appera in all metadata
        """

        operator = ")%20" + operator.upper() + "%20("
        if in_data == "all_meta":
            in_data = "%22All%20Metadata%22:"
        elif in_data == "author":
            in_data = "%22Author%20Keywords%22:"
        if url:
            self.url = url

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

    def _find_links(self):
        """goes through every paper on every page and collects all links to the
        subpages of the papers"""

        self.data = []
        document = "/document/"
        citation = "/citations"
        for link in self.linkstorage:
            if (document in link) & (citation not in link):
                self.data.append(link)
        # remove unnecessary results of the href search
        self.data = np.array(self.data)
        # remove duplicates that are in there due to multiple occurrence in the
        self.data = np.unique(self.data)
        print("Total number of abstracts that will be scraped:", len(self.data))

    def _find_abstracts(
        self,
        features=[
            "abstract",
            "title",
            "citationCount",
            "doi",
            "totalDownloads",
            "keywords",
            "publicationYear",
            "authors",
        ],
    ):
        """Opens all links for the webpages for each paper and scrapes the
        paper's abstract

        Args:
            features (list, optional): specify which information you want to
                have scraped. Defaults to ["abstract", "title", "citationCount", "doi",
                "totalDownloads", "keywords", "publicationYear", "authors"].
        """

        # initialize emtpy dict to create attributes and set dict keys as
        # features input and values as empty lists
        att_creator = {}
        for i in range(len(features)):
            att_creator[features[i]] = []

        # cretae the object attributes based on the input in list (features)
        for i, j in att_creator.items():
            setattr(self, i, j)

        # loop through number of  "link"
        for i in tqdm(range(len(self.data))):
            # only "try" because sometimes the javascript is corrupted
            # get the html data of the webpage with metadata such as abstracts
            # titles etc.
            try:
                data = json_loads(
                    re.search(
                        r"\.metadata=(.*?);",
                        requests_get(self.data[i]).text,
                    ).group(1)
                )
            except BaseException:
                pass

            # loop through list of desired features and extract information
            for i in features:
                # if the data is not available, e.g. the paper has no
                # citations, use None
                try:
                    temp = data[i]
                except BaseException:
                    temp = None

                # for views the data is differently extracted
                if i == "totalDownloads":
                    try:
                        temp = data["metrics"][i]
                    except BaseException:
                        temp = None

                if i == "keywords":
                    try:
                        temp = data[i][0]["kwd"]
                    except BaseException:
                        temp = None

                if i == "authors":
                    try:
                        temp = []
                        n_authors = len(data[i])
                        for j in range(n_authors):
                            temp.append(data[i][j]["name"])
                    except BaseException:
                        temp = None

                # append to obj attribute
                getattr(self, i).append(temp)

    async def get(
        self, url=None, keywords=None, operator="OR", pages=2, in_data="author"
    ):
        """Simply runs the self._open function that open ieeeXplore and seraches
            for paper urls.

        Args:
            url (str, optional): when the user specifies an own search query on
                IEEEXplore. Defaults to None.

            keywords (iist, optional): keywords that are searched for. Defaults
                to None.

            operator (str, optional): [description]. Defaults to "OR".

            pages (int, optional): Number of pages the algorithm iterates over.
                Defaults to 2.

            in_data (str, optional): "author" or "all_meta" whether to search in
                author keywords or all metadata. Defaults to "author".

        """
        await self._open(
            url=url, keywords=keywords, operator=operator, pages=pages, in_data=in_data
        )

    async def get_abstracts(
        self,
        url=None,
        keywords=None,
        operator="OR",
        pages=2,
        in_data="author",
        features=[
            "abstract",
            "title",
            "citationCount",
            "doi",
            "totalDownloads",
            "keywords",
            "publicationYear",
            "authors",
        ],
    ):
        """Function to scrape abstracts of scientific papers from the givin url.

        We used https://ieeexplore.ieee.org/search/advanced to generate a
        list like https://ieeexplore.ieee.org/search/searchresult.jsp?action=se
        arch&newsearch=true&matchBoolean=true&queryText=(%22Author%20Keywords%22
        :cotton)&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=Tru
        e&rowsPerPage=100&pageNumber=1
        with the search results.
        The abstracts of the papers listet on that list of search results will
        be stored in a .txt file with the givin file name.

        Args:
            url (str, optional): when the user specifies an own search query on
                IEEEXplore. Defaults to None.

            keywords (iist, optional): keywords that are searched for. Defaults
                to None.

            operator (str, optional): [description]. Defaults to "OR".

            pages (int, optional): Number of pages the algorithm iterates over.
                Defaults to 2.

            in_data (str, optional): "author" or "all_meta" whether to search in
                author keywords or all metadata. Defaults to "author".

            features (list, optional): which features should be scraped.
                Defaults to ["abstract", "title", "citationCount", "doi",
                "totalDownloads", "keywords", "publicationYear", "authors"].

        Returns:
            pd.DataFrame: DataFrame containing the paper abstracts and the
                selected features
        """

        await self._open(
            url=url, keywords=keywords, operator=operator, pages=pages, in_data=in_data
        )
        self._find_links()
        self._find_abstracts(features=features)

        data = pd.DataFrame()
        for i in features:
            data[i] = getattr(self, i)

        data = data.drop_duplicates(subset=["abstract", "title"])
        return data


if __name__ == "__main__":

    async def main():
        AS = AbstractScraper()
        data = await AS.get_abstracts(
            url="https://ieeexplore.ieee.org/search/searchresult.jsp?highlight=true&returnType=SEARCH&matchPubs=true&rowsPerPage=100&sortType=paper-citations&returnFacets=ALL&pageNumber=1",
            pages=1,
        )

        return data

    data = asyncio.get_event_loop().run_until_complete(main())

    # data.to_csv("cit_patents.csv")
