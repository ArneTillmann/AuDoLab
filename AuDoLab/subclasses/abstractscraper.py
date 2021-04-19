import re
import webbot
import time
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import json
import asyncio
from pyppeteer import launch


class AbstractScraper:
    """
    Abstractscraper - scrapes through IEEE Xplore and, depending on the given
    search query, scrapes the abstracts of scientific papers
    """

    def __init__(self):
        #self.web = webbot.Browser()
        #self.wait = 10
        self.first = "https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&newsearch=true&matchBoolean=true&"
        self.second = "&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1"

    async def _open(self, url=None, keywords=None, operator="OR", in_data="author",
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
        await self._open_pypeteer(url, keywords, operator, in_data, pages)

    async def _open_pypeteer(self, url=None, keywords=None, operator="OR", in_data="author",
    pages=2):
        self._create_url(url, keywords, operator, in_data)
        self.html_page = []
        print("The algorithm is iterating through", pages,
              "pages")
        browser = await launch()
        page = await browser.newPage()
        await page.goto(self.url, {'waitUntil' : 'networkidle0'})

        self.linkstorage = []

        elements = await page.querySelectorAll('a')
        for element in elements:
            self.linkstorage.append(await page.evaluate('(element) => element.href', element))
        for i in range(2, pages+1):
            self._create_new_url(i)
            await page.goto(self.url, {'waitUntil' : 'networkidle0'})
            elements = await page.querySelectorAll('a')
            for element in elements:
                self.linkstorage.append(await page.evaluate('(element) => element.href', element))
        await browser.close()

    def _add_page_number(self, page, position):
        self.url = self.url[:position+len("pageNumber=")] + str(page) + self.url[position+len("pageNumber=")+len(str(page-1)):]

    def _add_str_plus_page_number(self, page):
        if self.url[-1] =='?':
            self.url = self.url + "pageNumber=" + str(page)
        else:
            self.url = self.url + "?pageNumber" + str(page)

    def _create_new_url(self, page):

        if "pageNumber=" in self.url:
            self._add_page_number(page, self.url.find("pageNumber="))
        else:
            self._add_str_plus_page_number(page)

    def _create_url(self, url, keywords, operator, in_data):
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
        document_links = []
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
                        requests.get(self.data[i]).text,
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

    def get(self, url=None, keywords=None, operator="OR", pages=2, in_data="author"):
        self._open(
            url=url, keywords=keywords, operator=operator, pages=pages,
            in_data=in_data
        )

    async def get_abstracts(
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
        await self._open(
            url=url, keywords=keywords, operator=operator, pages=pages,
            in_data=in_data
        )
        self._find_links()
        self._find_abstracts()

        data = pd.DataFrame({"text": self.abstracts, "titles": self.title})
        data = data.drop_duplicates()
        return data




async def main():
    AS = AbstractScraper()
    data = await AS.get_abstracts(url=None, keywords=["dentistry", "teeth", "tooth"],
            in_data="all_meta",
            pages=3,
            operator="or")
    print(data)


asyncio.get_event_loop().run_until_complete(main())
