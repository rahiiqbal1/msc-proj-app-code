import os
import sys
from typing import Callable

from txtai import Embeddings
from PyQt5.QtWidgets import QApplication

from qt_gui import SearchWindow, SearchController
import transformer_get_results as ts
import txtai_bm25_get_results as txtaibm25
# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:
    # Directory where relevant data is stored:
    wikidataDir: str = os.path.join(os.pardir, os.pardir, "data")

    txtaiClassicalSearch(wikidataDir, "tfidf")

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
    SearchController(searchModelToUse, searchWindow, *argsOfSearchModelToUse)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

def transformerSearch(wikidataDir: str) -> None:
    """
    Gui search using transformer embeddings.
    """
    # Paths to ndjsons used for indexing and txtai embeddings:
    ndjsonsDir: str = os.path.join(wikidataDir, "poc-reduced-ndjsons")
    embeddingsPath: str = os.path.join(
        wikidataDir, "poc-txtai-embeddings-1"
    )

    # Loading embeddings:
    embeddings: Embeddings = dm.load_embeddings(embeddingsPath)

    guiSearch(ts.transformerGetResultsSF, ndjsonsDir, embeddings) 

def txtaiClassicalSearch(wikidataDir: str, searchMethod: str) -> None:
    """
    Gui search using txtai bm25/tf-idf implementation for results.

    searchMethod must be either "bm25" or "tfidf".
    """
    # Path to pickled jsons used for indexing:
    jsonsPickleSavePath: str = os.path.join(
        wikidataDir, "poc-fully-processed-ndjsons.pkl"
    )
    # Path to index:
    txtaiClassicalIndexPath: str = os.path.join(
        wikidataDir, f"poc_txtai_{searchMethod}_index.pkl"
    )

    guiSearch(
        txtaibm25.txtaiBM25GetResultsSF,
        jsonsPickleSavePath,
        txtaiClassicalIndexPath
    ) 

if __name__ == "__main__":
    main()
