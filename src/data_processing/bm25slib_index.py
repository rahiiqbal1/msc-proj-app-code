import sys
import os

import bm25s

import data_manipulation as dm

def main() -> None:
    # Directory storing all application data:
    all_data_dir: str = os.path.join(os.pardir, os.pardir, "data")
    # Path to pickled data which we want to index:
    pickled_data_path: str = os.path.join(
        all_data_dir, "poc-fully-processed-ndjsons.pkl"
    )
    # Directory in which to save index:
    index_save_dir: str = os.path.join(all_data_dir, "bm25s-index")

    # Fields within the data which we want to consider for the index:
    fields_to_use: tuple[str, ...] = ("name", "abstract")

    # Indexing:
    bm25slib_index_pickle(pickled_data_path, index_save_dir, fields_to_use)

    sys.exit(0)
    
def bm25slib_index_pickle(
    pickled_data_path: str,
    index_save_dir: str,
    fields_to_use: tuple[str, ...] 
    ) -> None:
    """
    Indexes data by loading in pickle at given path. Data must be a list of
    jsons, i.e. list[dict[str, str]].

    Indexes using the bm25s library.
    """
    # Loading in data, selecting only desired fields, and converting to
    # strings:
    json_data_as_strings_chosen_fields: list[str] = dm.stringify_dictionaries(
        dm.cut_list_of_dicts(
            dm.load_data(pickled_data_path), fields_to_use
        )
    )

    # Initialising BM25 model object and indexing data:
    bm25_model = bm25s.BM25()
    bm25_model.index(bm25s.tokenize(json_data_as_strings_chosen_fields))

    # Saving:
    bm25_model.save(index_save_dir)

if __name__ == "__main__":
    main()
