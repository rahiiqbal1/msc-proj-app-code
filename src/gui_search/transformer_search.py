import os
import sys

from txtai import Embeddings

from qt_gui import NUM_RESULTS_TO_SHOW
# Dirty hack!
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

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

