import os
import sys
import math

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
            wikidata_dir, pickled_jsons_dir, fields_to_use)

    # Looping through all of the data to index it:
    cut_jsons: list[dict[str, str]]
    for cut_jsons in tqdm(
            dm.generate_list_of_jsons_from_pickles(pickled_jsons_dir),
            total = NUMBER_OF_NDJSONS
        ):
        upsert_jsons_text_to_index(cut_jsons, index_save_path)

    sys.exit(0)

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

    for idx, single_json_text in enumerate(jsons_to_upsert_subset_as_strings):
        # Name for the current 'document' within the txtai index:
        json_doc_name: str = f"doc{idx}"
        embeddings.upsert(
            (json_doc_name, {"text": single_json_text}, None)
        )
        
    # Save:
    embeddings.save(index_save_path)

if __name__ == "__main__":
    main()
