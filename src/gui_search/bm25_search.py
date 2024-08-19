import os
import sys
import math

from nltk import word_tokenize

# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

# Model:
def bm25GetResultsSF(
    jsonData: list[dict[str, str]],
    dataIndex: dict[str, dict[int, int]],
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through given data using bm25 algorithm. Returns results with all
    fields intact, so they should be cut as desired outside of this function.

    Assumes that the json data is in the order in which it was indexed. Will 
    not function correctly if this is not the case.
    """
    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    return results

def _singleDocBM25Score(
    singleJsonIdxInColl: int,
    dataIndex: dict[str, dict[int, int]],
    searchQueryWords: list[str],
    avgDocLength: float,
    numDocsInCollection: int
    ) -> float:
    """
    Returns the bm25 score of a search query for a single document.
    """
    # Initialising variable to keep track of score, this will be added to 
    # sequentially as we calculate the sum for each word in the query:
    bm25Score: float = 0

    # Setting values of constants:
    k: float = 1.5
    b: float = 0.75

    queryWord: str
    for queryWord in searchQueryWords:
        # Getting the number of times that the current word appears in the
        # current document:
        numTimesWordInDoc: int = dataIndex[queryWord][singleJsonIdxInColl]

        # Getting the IDF for the current word:
        thisWordIDF: float = _singleWordIDF(
            queryWord, dataIndex, numDocsInCollection
        )

        bm25Numerator: float = numTimesWordInDoc * (k + 1)
        bm25Denominator: float = numTimesWordInDoc + k * (
            1 - b + (b * len(searchQueryWords)) / avgDocLength
        )

        bm25Score += thisWordIDF * (bm25Numerator / bm25Denominator)

    return bm25Score

def _singleWordIDF(
    word: str,
    dataIndex: dict[str, dict[int, int]],
    numDocsInCollection: int
    ) -> float:
    """
    Returns the IDF score for a single word with a given index and collection
    of data.
    """
    # Number of documents in the collection which contain the word:
    numDocsWithWord: int = len(dataIndex[word])

    return math.log(
        (
         (numDocsInCollection - numDocsWithWord + 0.5) /
         (numDocsWithWord + 0.5)
        ) + 1
    )
