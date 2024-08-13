import os
import sys
import math
import pickle
from typing import Any, Callable

from txtai import Embeddings
from PyQt5.QtWidgets import (
    QApplication
)

from qt_gui import NUM_RESULTS_TO_SHOW, SearchWindow, SearchController

NUM_ENTRIES = 6947320
PROPORTION_ENTRIES_TO_USE = 1

def main() -> None:
    use_poc_data(pocTransformerGetResults)

    sys.exit(0)

def use_full_data(model: Callable) -> None:
    """
    Uses any other specified data.
    """
    numEntriesUsed: int = math.floor(NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE)

    # Directory where relevant data is stored:
    wikidataDir: str = os.path.join(os.pardir, os.pardir, "data")

    # Get data and saved embeddings:
    entryJsons: list[dict[str, str]]
    embeddings: Embeddings
    entryJsons, embeddings = loadDataAndEmbeddings(
        os.path.join(wikidataDir, "poc_json_data.pkl"),
        os.path.join(wikidataDir, f"embeddings_subset_{numEntriesUsed}")
    )

    # Search using transformer:
    guiSearch(model,
              entryJsons,
              embeddings)

def use_poc_data(model: Callable) -> None:
    """
    Uses the data used for the proof of concept.
    """
    # Directory where relevant data is stored:
    wikidataDir: str = os.path.join(os.pardir, os.pardir, "data")

    # Get data and saved embeddings:
    entryJsons: list[dict[str, str]]
    embeddings: Embeddings
    entryJsons, embeddings = loadDataAndEmbeddings(
        os.path.join(wikidataDir, "poc_json_data.pkl"),
        os.path.join(wikidataDir, "poc_embeddings_subset_6947320")
    )

    # Search using transformer:
    guiSearch(model,
              entryJsons,
              embeddings)
        
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

# Model:
def pocTransformerGetResults(
    data: list[dict[str, str]],
    embeddings: Embeddings,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through given data using given transformer-model-derived
    embeddings. Returns results with all fields intact, so they should be cut
    as desired outside of this function.
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

def fullTransformerGetResults(
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

    return results

def loadDataAndEmbeddings(
    dataPath: str,
    embeddingsPath: str
    ) -> tuple[list[dict[str, str]], Embeddings]:
    '''
    Loads data with joblib (pickle) and it's associated txtai embeddings.

    Returns a tuple of data and embeddings.
    '''
    # Getting data:
    data: list[dict[str, str]] = loadData(dataPath)

    # Getting embeddings:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.load(embeddingsPath)

    return data, embeddings

def loadData(filePath: str) -> Any:
    '''
    Loads given object as python object using pickle.
    '''
    with open(filePath, "rb") as fileToRead:
        return pickle.load(fileToRead)

if __name__ == "__main__":
    main()
