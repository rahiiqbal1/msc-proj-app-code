import os
import sys
import math

from txtai import Embeddings
from tqdm import tqdm

import data_manipulation as dm

PROPORTION_ENTRIES_TO_USE = 1

def main() -> None:
    # Path to directory where all data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Path to pickled data:
    pickled_data_path: str = os.path.join(
        wikidata_dir, "poc-reduced-ndjsons.pkl"
    )

    # Path to save index at:
    index_save_path: str = os.path.join(
        wikidata_dir, f"poc-txtai-embeddings-{PROPORTION_ENTRIES_TO_USE}"
    )

    # Selecting fields we want to use to index the data:
    fields_to_use: tuple[str, ...] = ("name", "abstract")

    txtai_index_pickle(pickled_data_path, index_save_path, fields_to_use)

    sys.exit(0)

def txtai_index_pickle(
    pickled_data_path: str,
    index_save_path: str,
    fields_to_use: tuple[str, ...]
    ) -> None:
    """
    Indexes the data in the given pickled list of jsons (list[dict[str, str]]).

    Uses the transformer model specified within.
    """
    # Want to generate word embeddings using specified model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

    # Loading in data, selecting only desired fields, and converting to
    # strings:
    json_data_as_strings_chosen_fields: list[str] = dm.stringify_dictionaries(
        dm.cut_list_of_dicts(
            dm.load_data(pickled_data_path), fields_to_use
        )
    )

    # Index data:
    embeddings.index(tqdm(json_data_as_strings_chosen_fields))

    # Save index:
    embeddings.save(index_save_path)

def txtai_index_multi_pickle(
    wikidata_dir: str,
    ndjson_dir: str,
    pickled_jsons_dir: str,
    fields_to_use: tuple[str],
    index_save_path: str
    ) -> None:
    """
    Indexes the json data stored as separate pickled files in
    pickled_jsons_dir. Considers only fields_to_use in index, and saves at
    index_save_path.

    The files in ndjson_dir must correspond to those in pickled_jsons_dir. That
    is, each pickled file corresponds to each ndjson in the directory.
    """
    # Attempt to create the pickled data:
    if len(os.listdir(pickled_jsons_dir)) == 0:
        dm.create_pickled_cut_jsons(
            ndjson_dir, pickled_jsons_dir, fields_to_use
        )

    # Number ndjson to begin at. This allows us to index the data within the 
    # given directory in batches:
    start_position_save_name: str = "start_position.pkl"
    try:
        start_position: int = dm.load_data(
            os.path.join(wikidata_dir, start_position_save_name)
        )

    except FileNotFoundError:
        start_position: int = 0

    # Looping through all of the data to index it:
    cut_jsons: list[dict[str, str]]
    for idx, cut_jsons in enumerate(
            dm.generate_list_of_jsons_from_pickles(pickled_jsons_dir)
        ):
        # If we are not at the desired start position, continue to the next
        # iteration of the loop i.e. the next file in the list:
        if idx != start_position:
            continue

        txtai_upsert_jsons_text_to_index(cut_jsons, index_save_path)

        # Updating start position within list and saving:
        start_position += 1
        dm.save_data(start_position, start_position_save_name, wikidata_dir)

def txtai_upsert_jsons_text_to_index(
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

    # Loading previous state of saved index if it exists:
    if embeddings.exists(index_save_path):
        embeddings.load(index_save_path)

    # Indexing:
    embeddings.upsert(tqdm(jsons_to_upsert_subset_as_strings))

    print(embeddings.count())
        
    # Saving:
    embeddings.save(index_save_path)

if __name__ == "__main__":
    main()
