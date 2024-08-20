import os
import sys
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
    jsonDataDir: str = os.path.join(wikidataDir, "pickled-lists-of-cut-jsons")
    # jsonDataPath: str = os.path.join(wikidataDir, "poc_json_data.pkl")
    embeddingsPath: str = os.path.join(
        wikidataDir, "embeddings_subset_6947320"
    )
    
    transformerSearchMF(
        ts.transformerGetResultsMF, jsonDataDir, embeddingsPath
    )

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

def transformerSearchSF(
    model: Callable,
    jsonDataPath: str,
    embeddingsPath: str
    ) -> None:
    """
    Uses the pickled json data and the txtai index at the given paths.
    """

    # Get data and saved embeddings:
    entryJsons: list[dict[str, str]] = dm.load_data(jsonDataPath)
    embeddings: Embeddings = ts.loadEmbeddings(embeddingsPath)

    # Search using transformer:
    guiSearch(model, entryJsons, embeddings)
        
def transformerSearchMF(
    model: Callable,
    jsonDataDir: str,
    embeddingsPath: str
    ) -> None:
    """
    Uses the pickled json data in the given directory and the txtai index at
    the given path.
    """
    embeddings: Embeddings = ts.loadEmbeddings(embeddingsPath)

    guiSearch(model, jsonDataDir, embeddings)

if __name__ == "__main__":
    main()
