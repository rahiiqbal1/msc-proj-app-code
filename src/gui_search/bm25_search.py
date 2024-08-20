import os
import sys
import math
import copy

from nltk import word_tokenize

# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

# Model:
def bm25GetResultsSF(
    jsonData: list[dict[str, str]],
    dataIndex: dict[str, dict[int, int]],
    searchQuery: str,
    avgDocLength: float,
    numDocsInCollection: int,
    numResultsToShow: int
    ) -> list[dict[str, str]]:
    """
    Searches through given data using bm25 algorithm. Returns results with all
    fields intact, so they should be cut as desired outside of this function.

    Assumes that the json data is in the order in which it was indexed. Will 
    not function correctly if this is not the case.
    """
    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    # Getting jsons as strings to process:
    jsonsAsStrings: list[str] = dm.stringify_dictionaries(jsonData)

    # Getting bm25 scores for documents:
    bm25Scores: dict[int, float] = _collectionOfDocsBM25Scores(
        jsonsAsStrings,
        dataIndex,
        searchQuery,
        avgDocLength,
        numDocsInCollection
    )

    # Getting top scores along with the index of the documents which achieve
    # them:
    topIdxsAndScores: list[tuple[int, float]] = _sortScores(
        bm25Scores, numResultsToShow
    )

    # Getting relevant documents and adding to results:
    idxAndScore: tuple[int, float]
    for idxAndScore in topIdxsAndScores:
        results.append(jsonData[idxAndScore[0]])

    return results

def _sortScores(
    scores: dict[int, float],
    numScoresWanted: int
    ) -> list[tuple[int, float]]:
    """
    Sorts scores in the format of {data_index: score}. 

    Returns a list of (index, score) pairs.
    """
    # Copying input dictionary to prevent mutation:
    scoresCopy: dict[int, float] = copy.deepcopy(scores)

    # Initialising list to store sorted scores:
    sortedScores: list[tuple[int, float]] = []

    while len(sortedScores) != numScoresWanted:
        # Initialising variable to keep track of top score and it's index:
        topIdxAndScore: tuple[int, float] = (0, 0)

        docIdx: int
        for docIdx in scoresCopy:
            # If the score for the current document is higher than the tracked
            # top score, set it as the top score:
            if scoresCopy[docIdx] > topIdxAndScore[1]:
                topIdxAndScore = (docIdx, scoresCopy[docIdx])

        # Should have the top score in the dictionary, so add it to the list of
        # top scores:
        sortedScores.append(topIdxAndScore)

        # Remove the top score found this iteration from the dictionary to 
        # search again:
        del scoresCopy[topIdxAndScore[0]]

    return sortedScores

def _collectionOfDocsBM25Scores(
    documents: list[str],
    dataIndex: dict[str, dict[int, int]],
    searchQuery: str,
    avgDocLength: float,
    numDocsInCollection: int
    ) -> dict[int, float]:
    """
    Calculates the bm25 scores for a collection of documents.

    Returns a dictionary with the document index as key, and the bm25 score as
    value.
    """
    # Initialising dictionary to store bm25 scores:
    bm25Scores: dict[int, float] = {}

    # Looping over the index of each document in the collection:
    docIdxInColl: int
    for docIdxInColl in range(len(documents)):
        # Calculating bm25 score for current document and query and adding to 
        # dictionary:
        bm25Scores[docIdxInColl] = _singleDocBM25Score(
            docIdxInColl,
            dataIndex,
            word_tokenize(searchQuery),
            avgDocLength,
            numDocsInCollection
        )

    return bm25Scores

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
