import asyncio
import pandas as pd
from AuDoLab.subclasses import abstractscraper
from AuDoLab.subclasses import lda
from AuDoLab.subclasses import one_class_svm
from AuDoLab.subclasses import preprocessing
from AuDoLab.subclasses import tf_idf
from AuDoLab.subclasses import abstractscraper_arxiv
from AuDoLab.subclasses import abstractscraper_pubmed
import sys
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

# source: https://stackoverflow.com/questions/42009202/how-to-call-a-async-function-contained-in-a-class

class I3EGilli:

    def __init__(self):
        self.loop = asyncio.get_event_loop()

    def get_ieee(self, pages=10):
        return self.loop.run_until_complete(self.__async__get_ieee(
            "https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText=cotton&highlight=true&returnFacets=ALL&returnType=SEARCH&matchPubs=true&rowsPerPage=100&pageNumber=1",
            prepro=True,
            pages=pages,
        ))

    async def __async__get_ieee(
            self,
            url=None,
            keywords=None,
            operator="OR",
            pages=2,
            in_data="author",
            prepro=False,
            ngram_type=2,
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
                url (str, optional): [description]. Defaults to None.
                keywords (list, optional): List of keywords that are searched for. Defaults to None.
                operator (str, optional): Operator between the keywords. "AND" or "OR". If "AND" the search results must include all keywords. Defaults to "OR".
                pages (int, optional): Number of pages that are iterated over. Translates directly to number of abstracts that are scraped. Roughly there are 100 abstracts scraped per page. Defaults to 2.
                in_data (str, optional): If the keywords are searched for in the author keywords or in all metadata. Defaults to "author".
                prepro (bool, optional): if True, the scraped data will directly be preprocessed for later use. Defaults to False.
                ngram_type (int, optional): number of ngrams in preprocessing. Defaults to 2.

            Returns:
                pd.DataFrame: DataFrame with the stored abstracts and metadata
            """
            print('It is running')

            number = pages

            ks = abstractscraper.AbstractScraper()
            self.abstracts = await ks.get_abstracts(
                url=url, keywords=keywords, operator=operator, pages=number, in_data=in_data
            )

            if prepro == True:
                self.abstracts = self.abstracts.reset_index(drop=True)
                self.abstracts = self.text_cleaning(
                    self.abstracts, "abstract", ngram_type=ngram_type
                )

            if type(self.abstracts) != pd.DataFrame:
                print(
                    "if using the ieee abstractscraper, please use the following code: \n \n"
                    + "async def scrape():"
                    + "\n     return await audo.ieee_scraper(keywords=[keywords], prepro=False, pages=1, ngram_type=2)"
                    + "\n\nscraped_documents = asyncio.get_event_loop().run_until_complete(scrape())"
                )

                sys.exit(
                    "please specify the code as indicated above, or use the function abstract_scraper to scrape from different websites"
                )

            return await self.abstracts
    

from AuDoLab import AuDoLab
import asyncio
import nest_asyncio
nest_asyncio.apply()
from numpy import round as np_round
from numpy import arange as np_arange


audo = AuDoLab.AuDoLab()

if __name__ == "__main__":

    # Load target data from reuters dataset
    from nltk.corpus import reuters
    import numpy as np
    import pandas as pd

    data = []

    for fileid in reuters.fileids():
        tag, filename = fileid.split("/")
        data.append(
            (filename,
             ", ".join(
                 reuters.categories(fileid)),
                reuters.raw(fileid)))

    # store loaded data in dataframe
    data = pd.DataFrame(data, columns=["filename", "categories", "text"])

    #####------
    # start using audolab

    # clean theloaded data
    preprocessed_target = audo.text_cleaning(data=data, column="text")

    a = I3EGilli()
    a.get_ieee()
    print('It is done')