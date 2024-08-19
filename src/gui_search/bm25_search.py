import os
import sys
import math

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
    singleJson: dict[str, str],
    singleJsonIdxInColl: int,
    dataIndex: dict[str, dict[int, int]],
    searchQuery: str,
    avgDocLength: float
    ) -> float:
    """
    Returns the bm25 score of a search query for a single document.
    """
    return 1.0

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
