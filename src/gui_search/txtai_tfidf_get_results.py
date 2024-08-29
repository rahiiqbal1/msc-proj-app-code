import sys
import os

from txtai.scoring import ScoringFactory

from qt_gui import NUM_RESULTS_TO_SHOW
# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:

    sys.exit(0)

# Model:
def txtaiTFIDFGetResultsSF(
    jsonsPickleSavePath: str,
    indexSavePath: str,
    searchQuery: str
    ) -> list[dict[str, str]]:
    """
    Searches through jsons in given pickle using the txtai implementation of
    tf-idf. Returns result jsons with all fields intact, so they should be cut
    as desired outside of this function.
    """
    # Loading json data:
    jsonsIndexed: list[dict[str, str]] = dm.load_data(jsonsPickleSavePath)

    # Loading in txtai index:
    tfidfScoring = ScoringFactory.create({"method": "tfidf", "terms": True})
    tfidfScoring.load(indexSavePath)

    # Initialising list to store results:
    results: list[dict[str, str]] = []

    # Getting search results. This is a list of the form
    # [(id: int, score: float), ...]:
    allSearchResults = tfidfScoring.search(searchQuery, NUM_RESULTS_TO_SHOW)

    # There may be LSP errors here, but it works:
    for numericalResult in allSearchResults:
        results.append(jsonsIndexed[numericalResult[0]])

    return results

if __name__ == "__main__":
    main()
