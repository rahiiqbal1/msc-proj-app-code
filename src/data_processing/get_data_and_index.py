import os
import sys
import math
from typing import Any

from txtai import Embeddings
from tqdm import tqdm

import data_manipulation as dm

NUMBER_OF_NDJSONS = 372
PROPORTION_ENTRIES_TO_USE = 1

def main() -> None:
    # Path to directory where all data is stored:
    all_data_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Path to wikipedia data:
    wikidata_dir: str = os.path.join(all_data_dir, "reduced-nohtml-ndjsons")

    # Directory in which pickled lists of cut jsons are stored:
    pickled_jsons_dir: str = os.path.join(
        all_data_dir, "pickled-lists-of-cut-jsons"
    )

    # Path to save index at:
    index_save_path: str = os.path.join(
        all_data_dir,
        f"embeddings_subset_{dm.NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE}"
    )

    # Fields of the data which we want to use:
    fields_to_use: tuple[str, ...] = ("name", "abstract", "wikitext", "url")

    # Attempt to create the pickled data:
    if len(os.listdir(pickled_jsons_dir)) == 0:
        dm.create_pickled_cut_jsons(
            wikidata_dir, pickled_jsons_dir, fields_to_use
        )

    # Looping through all of the data to index it:
    cut_jsons: list[dict[str, str]]
    for cut_jsons in tqdm(
            dm.generate_list_of_jsons_from_pickles(pickled_jsons_dir),
            total = NUMBER_OF_NDJSONS
        ):
        upsert_jsons_text_to_index(cut_jsons, index_save_path)

    sys.exit(0)

    # # Full path of json data file:
    # json_data_save_path: str = os.path.join(
    #     all_data_dir, "pickled_json_data.pkl"
    # )
    # # Loading in json data:
    # all_jsons: list[dict[str, str]] = gen_or_get_pickled_jsons(
    #     wikidata_dir, json_data_save_path
    # )

    # # Using only num_entries_to_use entries to save computation time:
    # num_jsons_to_use: int = math.floor(
    #     NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE
    # )
    # subset_of_jsons: list[dict[str, str]] = all_jsons[
    #     : num_jsons_to_use
    # ]

    # subset_of_jsons_as_strings: list[str] = stringify_dictionaries(
    #     subset_of_jsons
    # )

    # # If the index does not already exist at the specified path, index and
    # # save:
    # index_save_path: str = (
    #     os.path.join(
    #         wikidata_dir,
    #         f"embeddings_subset_{num_jsons_to_use}"
    #     )
    # )
    # # Attempt to generate and save index:
    # gen_or_get_index(subset_of_jsons_as_strings, index_save_path)

def upsert_jsons_text_to_index(
    jsons_to_upsert: list[dict[str, str]],
    index_save_path: str
    ) -> None:
    """
    Takes a list of json objects as dictionaries and attempts to index their
    text, upserting into an index at the given save path.
    """
    # Using only the specified proportion of the data:
    num_jsons_to_use: int = math.floor(
        len(jsons_to_upsert) * PROPORTION_ENTRIES_TO_USE
    )
    jsons_to_upsert_subset: list[dict[str, str]] = jsons_to_upsert[
        : num_jsons_to_use
    ]

    # Converting each dictionary to single strings of their fields:
    jsons_to_upsert_subset_as_strings: list[str] = dm.stringify_dictionaries(
        jsons_to_upsert_subset
    )

    # Want to generate word embeddings using specified hugging-face model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

    # Index:
    embeddings.upsert(tqdm(jsons_to_upsert_subset_as_strings))
    # Save:
    embeddings.save(index_save_path)

# Old.
def gen_or_get_index(data_to_index: Any, index_save_path: str) -> None:
    '''
    Indexes the given data, and saves the index at the given path. Uses
    the model in code by default.
    '''
    # Want to generate word embeddings using specified hugging-face model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

    if os.path.isfile(index_save_path) == False:
        # Index data:
        embeddings.index(tqdm(data_to_index))
        # Save index:
        embeddings.save(index_save_path)
    else:
        print("An index is already saved at the given path.")

# Old.
def gen_or_get_pickled_jsons(
    ndjson_data_dir: str,
    data_save_path: str
    ) -> list[dict[str, str]]:
    '''
    If the file at the given path already exists, attempt to load it using
    pickle. If not, read all .ndjson files in the given directory and attempt
    to read set fields from each .json within them.

    Returns a list of all .json entries in the data as dictionaries.
    '''
    # Fields of the data which we want to use:
    fields_to_use: tuple[str, ...] = ("name", "abstract", "Categories", "url")

    # If the data does not exist, proceed to generate it. Otherwise load it:
    if os.path.isfile(data_save_path) == False:
        # Initialising list to store dictionaries with desired fields:
        all_cut_entry_jsons: list[dict[str, str]] = []

        # List of deserialised json objects per .ndjson file read:
        entry_jsons: list[dict[str, Any]]
        for entry_jsons in dm.generate_jsons_from_ndjsons(ndjson_data_dir):
            # Reducing all dictionaries in the current list to the chosen 
            # fields and adding to the list to store them all:
            all_cut_entry_jsons += dm.cut_list_of_dicts(
                entry_jsons, fields_to_use
            )

        # Saving data:
        data_save_dir: str
        data_save_name: str
        data_save_dir, data_save_name = os.path.split(data_save_path)
        dm.save_data(all_cut_entry_jsons, data_save_name, data_save_dir)   

        return all_cut_entry_jsons

    else:
        # Loads using pickle, so the file must be deserialisable as the correct
        # return type.
        return dm.load_data(data_save_path)

if __name__ == "__main__":
    sys.exit(main())
