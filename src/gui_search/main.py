import os
import sys
import math
from typing import Callable

from txtai import Embeddings
from PyQt5.QtWidgets import QApplication

from qt_gui import SearchWindow, SearchController
import transformer_search as ts
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
    
    transformerSearch(ts.transformerGetResultsSF, jsonDataPath, embeddingsPath)

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
    entryJsons, embeddings = ts.loadDataAndTxtaiEmbeddings(
         jsonDataPath, embeddingsPath
    )

    # Search using transformer:
    guiSearch(model,
              entryJsons,
              embeddings)
        
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

if __name__ == "__main__":
    main()
