import sys
import os

from bm25s import BM25, tokenize
from numpy import ndarray

from qt_gui import NUM_RESULTS_TO_SHOW
# Bodge
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:

    sys.exit(0)

def bm25slibGetResultsSF(
    ndjsonsDir: str,
    bm25sIndex: BM25,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through guven data using given bm25s (library) index. 

    Returns results with all fields intact, so they should be cut as desired
    outside of this function.
    """
    # Getting results indices:
    resultIndices: ndarray
    resultIndices, _ = bm25sIndex.retrieve(
        tokenize(searchQuery), k = NUM_RESULTS_TO_SHOW
    )

    # Initialising list to store readable results:
    results: list[dict[str, str]] = []

    # Getting results in readable format:
    resultIdx: int
    for resultIdx in resultIndices:
        readableResult: dict[str, str] = dm.find_json_given_index_ndjsons(
            ndjsonsDir, resultIdx
        )
        results.append(readableResult)

    return results

if __name__ == "__main__":
    main()
