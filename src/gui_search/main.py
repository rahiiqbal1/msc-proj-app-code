import os
import sys
import math
from typing import Callable

from txtai import Embeddings
from PyQt5.QtWidgets import (
    QApplication
)

from qt_gui import NUM_RESULTS_TO_SHOW, SearchWindow, SearchController
# Dirty hack!
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:
    # Directory where relevant data is stored:
    wikidataDir: str = os.path.join(os.pardir, os.pardir, "data")

    # Paths to desired pickled json data and txtai embeddings:
    jsonDataPath: str = os.path.join(wikidataDir, "poc_json_data.pkl")
    embeddingsPath: str = os.path.join(
        wikidataDir, "embeddings_subset_6947320"
    )
    
    transformerSearch(transformerGetResultsSF, jsonDataPath, embeddingsPath)

    sys.exit(0)

def guiSearch(
    searchModelToUse: Callable,
    *argsOfSearchModelToUse
    ) -> None:
    """
    Search through list of JSON data as python dictionaries. Uses GUI
    interface.
    """
    # Initialising app:
    searchApp = QApplication([])
    # Initialising window:
    searchWindow = SearchWindow()

    # Creating controller. Does not need to be stored as a variable as it holds
    # references to the model and view. Note that guiSearch takes all
    # arguments of the searchModel except the search query, which must be taken
    # from the searchWindow and so is not known initially. This is taken care
    # of within the SearchController:
    SearchController(
        searchModelToUse,
        searchWindow,
        *argsOfSearchModelToUse
    )

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

def transformerSearch(
    model: Callable,
    jsonDataPath: str,
    embeddingsPath: str
    ) -> None:
    """
    Uses the pickled json data and the txtai index at the given paths.
    """

    # Get data and saved embeddings:
    entryJsons: list[dict[str, str]]
    embeddings: Embeddings
    entryJsons, embeddings = loadDataAndTxtaiEmbeddings(
         jsonDataPath, embeddingsPath
    )

    # Search using transformer:
    guiSearch(model,
              entryJsons,
              embeddings)
        
# Model:
def transformerGetResultsSF(
    data: list[dict[str, str]],
    embeddings: Embeddings,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through given data using given transformer-model-derived
    embeddings. Returns results with all fields intact, so they should be cut
    as desired outside of this function.

    Assumes the data is stored within a single deserialised json object.
    """
    # This is a list of tuples containing the index of the result in the
    # data and it's score:
    numResults: list[tuple[int, float]] = embeddings.search(
        searchQuery, NUM_RESULTS_TO_SHOW
    )

    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    # Getting results in readable format:
    numResult: tuple[int, float]
    for numResult in numResults:
        readableResult: dict[str, str] = data[numResult[0]]
        results.append(readableResult)

    return results

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

# Model:
def transformerGetResultsMF(
    dataDir: str,
    embeddings: Embeddings,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through all (pickled) data within directory using given
    transformer-derived embeddings. Returns results with all fields intact, so
    they should be cut as desired outside of this function.

    Assumes that the data has been stored in the directory in numeric order and
    that the embeddings have been indexed in this order. Will not work properly
    if this is not the case.
    """
    # This is a list of tuples containing the index of the result in the
    # data and it's score:
    numResults: list[tuple[int, float]] = embeddings.search(
        searchQuery, NUM_RESULTS_TO_SHOW
    )

    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    # Getting results in readable format:
    numResult: tuple[int, float]
    for numResult in numResults:
        # Getting index of result in the search-index in order to know from 
        # where to retrieve the data:
        resultIndex: int = numResult[0]
        
        # Getting result from directory:
        results.append(findDataGivenIndex(dataDir, resultIndex))

    return results

def loadDataAndTxtaiEmbeddings(
    dataPath: str,
    embeddingsPath: str
    ) -> tuple[list[dict[str, str]], Embeddings]:
    '''
    Loads data with joblib (pickle) and it's associated txtai embeddings.

    Returns a tuple of data and embeddings.
    '''
    # Getting data:
    data: list[dict[str, str]] = dm.load_data(dataPath)

    # Getting embeddings:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.load(embeddingsPath)

    return data, embeddings

def findDataGivenIndex(dataDir: str, index: int) -> dict[str, str]:
    """
    For a directory containing numerically-sorted pickled lists of data, and a
    given index, return the data point from within the file specified by that
    index.
    """
    # Retrieving filenames within directory and sorting:
    sortedFilenames: list[str] = dm.sort_filenames_with_numbers(
        os.listdir(dataDir)
    )

    # Getting lengths of files:
    fileListLengths: list[int] = []
    filename: str
    for filename in sortedFilenames:
        # Getting full filepath as filename is only the name:
        filepathFull: str = os.path.join(dataDir, filename)

        # Loading data and adding it's length to the list of lengths:
        fileListLengths.append(
            len(dm.load_data(filepathFull))
        )

    # Initialising variable to keep track of which file within the directory
    # we are in the scope of through the for loop. Starting from 0 as we can 
    # use the list of sorted filenames to load the file:
    batchCounter: int = 0
    # Searching through lengths till the index is contained:
    listLength: int
    for listLength in fileListLengths:
        if index - listLength >= 0:
            index -= listLength
            batchCounter += 1

        else:
            break

    # Full path to file which contains the desired data point: 
    containingFilePath: str = os.path.join(
        dataDir, sortedFilenames[batchCounter]
    )

    # Loading correct data in directory and getting json at required index:
    return dm.load_data(containingFilePath)[index]

def testFindDataGivenIndex() -> None:
    testPickledListsDir: str = os.path.join(
        os.pardir, os.pardir, "data", "test-pickled-list-retrieval"
    )
    testIndex: int = 7

    listsToPickle: tuple[list[int], ...] = (
        [1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]
    )

    # Pickling data:
    listToPickle: list[int]
    for fileIdx, listToPickle in enumerate(listsToPickle):
        dm.save_data(listToPickle, f"{fileIdx}.pkl", testPickledListsDir)

    print(findDataGivenIndex(testPickledListsDir, testIndex))

if __name__ == "__main__":
    main()
