import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from data_manipulation import (
    load_ndjsons_as_single_list,
    stringify_dictionaries,
    save_data
)

def main() -> None:
    # Directory in which application data is stored:
    wikidata_dir: str = os.path.join(os.pardir, os.pardir, "data")

    # Directory containing .ndjson which we want to use as our corpus:
    ndjsons_dir: str = os.path.join(
        wikidata_dir, "poc-fully-processed-ndjsons"
    )

    # Loading in data from ndjsons as dictionaries and converting it to
    # strings:
    json_data_as_strings: list[str] = stringify_dictionaries(
        load_ndjsons_as_single_list(ndjsons_dir)
    )

    # Getting tfidf matrix and saving it:
    save_data(
        get_corpus_tfidf(json_data_as_strings),
        "poc-fully-processed-ndjsons-tfidf.pkl",
        wikidata_dir
    )

    sys.exit(0)

def get_corpus_tfidf(corpus: list[str]):
    """
    Returns a sparse matrix of TF-IDF features for the given corpus of text 
    with shape (n_samples, n_features).

    Each element in the list must be a single document.
    """
    vectoriser = TfidfVectorizer()

    return vectoriser.fit_transform(tqdm(corpus))

def test_get_corpus_tfidf() -> None:
    corpus = ["a line of text", "another piece of textual information"]

    print(get_corpus_tfidf(corpus))


if __name__ == "__main__":
    main()
