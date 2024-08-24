import sys
import os
from typing import Any

from txtai.scoring import ScoringFactory

from qt_gui import NUM_RESULTS_TO_SHOW
# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:

    sys.exit(0)

# Model:
def txtaiBM25GetResultsSF(
    jsonsPickleSavePath: str,
    indexSavePath: str,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through jsons in given pickle using the txtai implementation of
    BM25. Returns result jsons with all fields intact, so they should be cut as
    desired outside of this function.
    """
    # Loading json data:
    jsonsIndexed: list[dict[str, str]] = dm.load_data(jsonsPickleSavePath)

    # Loading in txtai index:
    bm25Scoring = ScoringFactory.create({"method": "bm25", "terms": True})
    bm25Scoring.load(indexSavePath)

    # Initialising list to store results:
    results: list[dict[str, str]] = []

    # Getting search results. This is a list of the form
    # [{"id": int, "text", str, "score", float}, ...]:
    allSearchResults = bm25Scoring.search(searchQuery, NUM_RESULTS_TO_SHOW)

    for result in allSearchResults:
        results.append(jsonsIndexed[result["id"]])

    return results

if __name__ == "__main__":
    main()
