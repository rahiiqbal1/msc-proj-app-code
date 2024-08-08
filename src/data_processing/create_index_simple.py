import sys
import os
import pickle
from remove_html import load_jsons_from_ndjson, generate_jsons_from_ndjsons

def main() -> None:
    # Directory in which the .ndjson files to be indexed are stored. Here we
    # use the files which have already had their unrelated fields and html 
    # removed, as well as their main body being preprocessed (lower case,
    # stopwords removed, lemmatised, punctuation removed):
    ndjson_store_dir: str = os.path.join(os.pardir, "reduced-nohtml-ndjsons")

    sys.exit(0)

def index_single_ndjson() -> None:
    """
    Creates index for a single NDJSON file, i.e. indexes all JSON files within.

    Saves index as pickled object.
    """

if __name__ == "__main__":
    main()
