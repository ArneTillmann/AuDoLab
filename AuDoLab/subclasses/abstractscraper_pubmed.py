import warnings
from bs4 import BeautifulSoup
import requests
import pandas as pd
import unicodedata
from tqdm import tqdm


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class AbstractScraper_Pubmed:
    def __init__(self):
        pass

    def _find_links(self, url, number_of_pages):
        """finds paper links associated with given url search query

        Args:
            url (str): Link from arxiv.org with search query

            number_of_pages (int): number of pages that should be scraped.
        """

        html = requests.get(url).text

        # get urls of pages staring from 1
        self.pages = [url + "&page=" + str(i) for i in range(1, number_of_pages + 1)]

        self.pages = list(set(self.pages))

        self.document_links = []

        # print statement so everybody knows what is happening
        if len(self.pages) == 1:
            print("The algorithm is iterating through", len(self.pages), "page")
        else:
            print("The algorithm is iterating through", len(self.pages), "pages")

        # loop through all pages and extract links
        for i in tqdm(range(len(self.pages))):
            html = requests.get(self.pages[i]).text

            for link in BeautifulSoup(html, features="html.parser").findAll(
                "a", {"class": "docsum-title"}
            ):
                self.document_links.append(
                    "https://pubmed.ncbi.nlm.nih.gov" + link["href"]
                )

        print(
            "The algorithm found ",
            len(self.document_links),
            " unique abstracts to in your query. \n Try increasing the number of pages if you want to scrape more papers",
        )

    def _scrape(self, author=True):
        """loop through all found paper links and scrape abstracts, titles and
        authors.

        Args:
            author (bool, optional): if true all author names are scraped as well. Defaults to True.
        """

        # initiliaze empty lists
        self.abstracts = []
        self.titles = []
        self.authors = []

        # loop through links and extract abstracts, titles and authors
        for url in tqdm(self.document_links):
            # for every link get underlying html code
            html = requests.get(url).text
            soup = BeautifulSoup(html, "html.parser")

            # use try (without specifiyng exception) for adapting to changes in
            # source data
            # extract all abstracts with beautifulSoup
            try:
                abstract = soup.find("div", class_="abstract-content selected")
                # extract the text between <span>
                abstract = abstract.text.replace("\n", "")
                # append to data
                self.abstracts.append(abstract)
            except BaseException:
                self.abstracts.append(None)

            try:
                title = soup.find("h1", class_="heading-title")
                title = title.text.replace("\n", "")
                self.titles.append(title)
            except BaseException:
                self.titles.append(None)

            if author == True:
                try:
                    authors = soup.find("div", class_="authors")
                    authors = authors.text.replace("\n", "")
                    authors = unicodedata.normalize("NFKD", authors)
                    authors = authors.replace("  ", "")
                    authors = "".join(i for i in authors if not i.isdigit())
                    self.authors.append(authors)
                except BaseException:
                    self.authors.append(None)

                data = pd.DataFrame(
                    {
                        "abstract": self.abstracts,
                        "title": self.titles,
                        "authors": self.authors,
                    }
                )
            else:
                data = pd.DataFrame(
                    {
                        "abstract": self.abstracts,
                        "title": self.titles,
                    }
                )

        self.data = data.drop_duplicates()

    def scrape_pubmed(self, url, pages=8, author=True):
        """Scrapes https://pubmed.ncbi and returns a pd.DataFrame containing
            abstracts, titles and author names.

        Returns these informations based on the users given url (search query),
        e.g. url="https://pubmed.ncbi.nlm.nih.gov/?term=medicine"

        Args:
            url (string): link of searchquery from arxiv.org

            pages (int, optional): number of pages the algorithm iterates through and searches for abstracts. Defaults to 8.

            author (bool, optional): if true all author names are scraped as well. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame that contains: Abstracts, Titles and Authors
        """

        if "pubmed" not in url:
            return print(
                "ERROR: Only specify a url/search query via https://pubmed.ncbi"
            )
        else:
            self._find_links(url=url, number_of_pages=pages)
            self._scrape(author)

        return self.data


if __name__ == "__main__":
    AS = AbstractScraper_Pubmed()

    test = AS.scrape_pubmed("https://pubmed.ncbi.nlm.nih.gov/?term=medicine", 3, False)
