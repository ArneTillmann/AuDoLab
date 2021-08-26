import warnings
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from tqdm import tqdm


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class AbstractScraper_Arxiv:
    def __init__(self):
        pass

    def _find_links(self, url, number_of_pages):
        """finds paper links associated with given url search query

        Args:
            url (str): Link from arxiv.org with search query

            number_of_pages (int): number of pages that should be scraped
        """

        html = requests.get(url).text

        self.pages = []

        for page in BeautifulSoup(html, features="html.parser").findAll(
            "a", {"class": "pagination-link"}
        ):
            self.pages.append("https://arxiv.org" + page["href"])

        self.pages = list(set(self.pages))

        if len(self.pages) > number_of_pages:
            self.pages = self.pages[:number_of_pages]
        else:
            self.pages = self.pages

        self.document_links = []

        if len(self.pages) == 1:
            print("The algorithm is iterating through", len(self.pages), "page")
        else:
            print("The algorithm is iterating through", len(self.pages), "pages")

        for i in tqdm(range(len(self.pages))):
            html = requests.get(self.pages[i]).text

            for link in BeautifulSoup(html, features="html.parser").find_all(
                "a", href=re.compile("/abs")
            ):
                self.document_links.append(link.get("href"))

        print(
            "The algorithm found ",
            len(self.document_links),
            " unique abstracts in your query. \n Try increasing the number of pages if you want to scrape more papers",
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

            # use try (without specifiyng exception) for adapting to changes in source data
            # extract all abstracts with beautifulSoup
            try:
                abstract = soup.find("blockquote", class_="abstract")
                # extract the text between <span>
                abstract.span.extract()
                # replace \n (new lines)
                abstract = abstract.text.replace("\n", "")
                # append to data
                self.abstracts.append(abstract)
            except BaseException:
                self.abstracts.append(None)

            try:
                title = soup.find("h1", class_="title")
                title.span.extract()
                self.titles.append(title.text)
            except BaseException:
                self.titles.append(None)

            if author == True:
                try:
                    authors = soup.find("div", class_="authors")
                    authors.span.extract()
                    self.authors.append(authors.text)
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

    def scrape_arxiv(self, url, pages=8, author=True):
        """Scrapes arxiv.org and returns a pd.DataFrame containing abstracts,
            titles and author names.

        Returns these informations based on the users given url (search query),
        e.g. url="https://arxiv.org/search/?query=deep+learning&searchtype=all&source=header&order=&size=100&abstracts=show&date-date_type=submitted_date&start=0"

        Args:
            url (str): link of searchquery from arxiv.org

            pages (int, optional): number of pages the algorithm iterates
                through and searches for abstracts. Defaults to 8.

            author (bool, optional): if true all author names are scraped as well. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame that contains Abstracts, Titles and Authors
        """

        if "arxiv" not in url:
            return print("ERROR: Only specify a url/search query via arxiv.org")
        else:
            self._find_links(url=url, number_of_pages=pages)
            self._scrape(author)

        return self.data


if __name__ == "__main__":
    AS = AbstractScraper_Arxiv()

    test = AS.scrape_arxiv(
        "https://arxiv.org/search/?query=deep+learning&searchtype=all&source=header&order=&size=100&abstracts=show&date-date_type=submitted_date&start=0",
        1,
    )
