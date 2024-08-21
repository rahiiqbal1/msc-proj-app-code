import os
import sys
from typing import Callable

from txtai import Embeddings
from PyQt5.QtWidgets import QApplication

from qt_gui import SearchWindow, SearchController, NUM_RESULTS_TO_SHOW
import transformer_search as ts
import bm25_search as bm25
# Dirty hack!
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:
    # Directory where relevant data is stored:
    wikidataDir: str = os.path.join(os.pardir, os.pardir, "data")

    # # Paths to desired pickled json data and txtai embeddings:
    # jsonDataPath: str = os.path.join(wikidataDir, "poc_json_data.pkl")
    # embeddingsPath: str = os.path.join(
    #     wikidataDir, "poc_embeddings_subset_6947320"
    # )

    # # Loading json data and embeddings:
    # documentJsons: list[dict[str, str]] = dm.load_data(jsonDataPath)
    # embeddings: Embeddings = dm.load_embeddings(embeddingsPath)

    # Path to .ndjson which was used to index and which we will read from:
    ndjsonPath: str = os.path.join(
        wikidataDir, "poc_combined_processed.ndjson"
    )
    # Path to index:
    indexPath: str = os.path.join(wikidataDir, "poc_simple.index")

    # Average document length in the given ndjson:
    avgDocLength: float = 43.42
    
    # Number of documents (jsons) in the given ndjson:
    numDocsInCollection: int = 2

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

if __name__ == "__main__":
    main()
