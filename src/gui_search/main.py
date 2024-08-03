import os
import sys
import math
from typing import Any, Callable
import pickle
from txtai import Embeddings
from PyQt5.QtWidgets import (
    QApplication
)
from qt_gui import SearchWindow, SearchController

NUM_ENTRIES = 6947320
PROPORTION_ENTRIES_TO_USE = 1

def main() -> int:
    num_entries_used: int = math.floor(NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE)

    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "wikidata"
    )

    # Get data and saved embeddings:
    entry_jsons: list[dict[str, str]]
    embeddings: Embeddings
    entry_jsons, embeddings = load_data_and_embeddings(
        os.path.join(wikidata_dir, "entry_data.gz"),
        os.path.join(wikidata_dir, f"embeddings_subset_{num_entries_used}")
    )

    gui_search(transformer_get_results,
               entry_jsons,
               embeddings)

    return 0
        
def gui_search(
    search_model_to_use: Callable,
    *args_of_search_model_to_use
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
    # references to the model and view:
    SearchController(search_model_to_use,
                     searchWindow,
                     *args_of_search_model_to_use)

    # Display and event loop:
    searchWindow.show()
    sys.exit(searchApp.exec())

def transformer_get_results(
    data: list[dict[str, str]],
    embeddings: Embeddings
    ) -> list[dict[str, str]]:
    """
    Searches through given data using given transformer-model-derived
    embeddings. Returns results with all fields intact, so they should be cut
    as desired outside of this function.
    """
    search_query: str = "test query"

    # This is a list of tuples containing the index of the result in the
    # data and it's score:
    num_results: list[tuple[int, float]] = embeddings.search(
        search_query, 10
    )

    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    # Getting results in readable format:
    num_result: tuple[int, float]
    for num_result in num_results:
        readable_result: dict[str, str] = data[num_result[0]]
        results.append(readable_result)

    return results

def load_data_and_embeddings(
    data_path: str,
    embeddings_path: str
    ) -> tuple[list[dict[str, str]], Embeddings]:
    '''
    Loads data with joblib (pickle) and it's associated txtai embeddings.

    Returns a tuple of data and embeddings.
    '''
    # Getting data:
    data: list[dict[str, str]] = load_data(data_path)

    # Getting embeddings:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.load(embeddings_path)

    return data, embeddings

def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using joblib.
    '''
    with open(file_path, "rb") as file_to_load:
        return pickle.load(file_to_load)

if __name__ == "__main__":
    sys.exit(main())
