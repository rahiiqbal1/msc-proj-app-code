import sys
import os

from txtai.scoring import ScoringFactory
from tqdm import tqdm

import data_manipulation as dm

def main() -> None:
    # Directory in which all application data is stored:
    wikidata_dir: str = os.path.join(os.pardir, os.pardir, "data")

    # Path to pickled json data which we want to index:
    pickled_jsons_path: str = os.path.join(
        wikidata_dir, "poc-fully-processed-ndjsons.pkl"
    )

    # Full path which we want to save the index at:
    index_save_path: str = os.path.join(
        wikidata_dir, "poc_txtai_bm25_index.pkl"
    )

    # Fields within the jsons which we want to consider for the index:
    fields_to_use: tuple[str, ...] = ("name", "abstract")

    # Indexing and saving:
    txtai_bm25_index_pickle(pickled_jsons_path, index_save_path, fields_to_use)

    sys.exit(0)

def txtai_bm25_index_pickle(
    pickled_jsons_path: str,
    index_save_path: str,
    fields_to_use: tuple[str, ...]
    ) -> None:
    """
    Indexes the pickled data at the given path.

    Data must be of the form list[dict[str, str]].
    """
    # Loading in data, selecting only desired fields, and converting to
    # strings:
    jsons_as_strings_cut: list[str] = dm.stringify_dictionaries(
        dm.cut_list_of_dicts(
            dm.load_data(pickled_jsons_path), fields_to_use
        )
    )

    # Indexing:
    bm25_scoring = ScoringFactory.create({"method": "bm25", "terms": True})
    bm25_scoring.index(
        tqdm(((i, text, None) for i, text in enumerate(jsons_as_strings_cut)),
             total = len(jsons_as_strings_cut)
        )
    )
    # Saving. Saves as a pickle:
    bm25_scoring.save(index_save_path)

if __name__ == "__main__":
    main()
