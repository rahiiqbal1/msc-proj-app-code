import json
import os
import sys
import math
import pickle
from typing import Any, Generator

from txtai import Embeddings
from tqdm import tqdm

NUM_ENTRIES = 6947320
PROPORTION_ENTRIES_TO_USE = 1

def main() -> int:
    # Path to directory where all data is stored:
    all_data_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Get path to wikipedia data:
    wikidata_dir: str = os.path.join(all_data_dir, "reduced-nohtml-ndjsons")

    # Full path of json data file:
    json_data_save_path: str = os.path.join(
        all_data_dir, "pickled_json_data.pkl"
    )
    # Loading in json data:
    all_jsons: list[dict[str, str]] = gen_or_get_pickled_jsons(
        wikidata_dir, json_data_save_path
    )

    # Using only num_entries_to_use entries to save computation time:
    num_jsons_to_use: int = math.floor(
        NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE
    )
    subset_of_jsons: list[dict[str, str]] = all_jsons[
        : num_jsons_to_use
    ]

    subset_of_jsons_as_strings: list[str] = stringify_dictionaries(
        subset_of_jsons
    )

    # If the index does not already exist at the specified path, index and
    # save:
    index_save_path: str = (
        os.path.join(
            wikidata_dir,
            f"embeddings_subset_{num_jsons_to_use}"
        )
    )
    # Attempt to generate and save index:
    gen_or_get_index(subset_of_jsons_as_strings, index_save_path)

    return 0

def stringify_dictionaries(
    dicts_to_stringify: list[dict[str, str]]
    ) -> list[str]:
    '''
    Takes a list of dictionaries which has both keys and values as string data,
    and returns a list of each field in the dictionary as a single combined
    string.
    '''
    strings_to_return: list[str] = []

    single_dict: dict[str, str]
    for single_dict in tqdm(dicts_to_stringify):
        # Initialising string to add to return list:
        string_for_return: str = ""
        dict_key: str 
        for dict_key in single_dict:
            string_for_return += single_dict[dict_key]

        strings_to_return.append(string_for_return)

    return strings_to_return

def load_jsons_from_ndjson(ndjson_file_path: str) -> list[dict[str, Any]]:
    '''
    Loads in json objects from a .ndjson file. Stores as python dictionaries in
    a python list.
    '''
    # Initialising list to store json objects:
    json_list: list[dict[str, Any]] = []
    # Opening file:
    with open(ndjson_file_path, 'r') as ndjson_file:
        # Reading .ndjson line-by-line, converting each line (json) to python
        # dictionary, and storing in list:
        json_line: str
        try:
            for json_line in ndjson_file:
                json_list.append(json.loads(json_line))
        except json.decoder.JSONDecodeError:
            pass

    return json_list

def generate_jsons_from_ndjsons(
    ndjsons_dir: str
    ) -> Generator[list[dict[str, Any]], None, None]:
    '''
    Acts as generator, yielding at each step a list of all deserialised json
    objects from a given file.
    '''
    # Get all filenames in ndjsons_dir to read through them:
    ndjson_filenames: list[str] = os.listdir(ndjsons_dir)

    ndjson_filename: str
    for ndjson_filename in tqdm(ndjson_filenames, "Reading .ndjsons"):
        # Getting full filepath as ndjson_filename is only the filename:
        ndjson_filepath: str = os.path.join(ndjsons_dir, ndjson_filename)

        # Yielding list of json dictionaries from the current file:
        yield load_jsons_from_ndjson(ndjson_filepath)

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
        for entry_jsons in generate_jsons_from_ndjsons(ndjson_data_dir):
            # Reducing all dictionaries in the current list to the chosen 
            # fields and adding to the list to store them all:
            all_cut_entry_jsons += cut_list_of_dicts(
                entry_jsons, fields_to_use
            )

        # Saving data:
        data_save_dir: str
        data_save_name: str
        data_save_dir, data_save_name = os.path.split(data_save_path)
        save_data(all_cut_entry_jsons, data_save_name, data_save_dir)   

        return all_cut_entry_jsons

    else:
        # Loads using pickle, so the file must be deserialisable as the correct
        # return type.
        return load_data(data_save_path)

def cut_list_of_dicts(
    list_of_dicts_to_cut: list[dict[str, Any]],
    fields_to_use: tuple[str, ...]
    ) -> list[dict[str, str]]:
    """
    Cuts all dictionaries in a given list.

    Returns a list of dictionaries with only the chosen fields of the 
    dictionaries from the given list. Chosen fields must have string type 
    values.
    """
    # initialising list to store cut dictionaries:
    list_of_cut_dicts: list[dict[str, str]] = []

    single_dict: dict[str, Any]
    for single_dict in list_of_dicts_to_cut:
        single_cut_dict: dict[str, str] = cut_single_dict(
            single_dict, fields_to_use
        )

        list_of_cut_dicts.append(single_cut_dict)

    return list_of_cut_dicts

def cut_single_dict(
    dict_to_cut: dict[str, Any],
    fields_to_use: tuple[str, ...]
    ) -> dict[str, Any]:
    """
    Selects only the chosen fields from the given dictionary and returns the
    reduced version.

    Returned dictionary must have both keys and values as strings.
    """
    cut_dict_for_return: dict[str, str] = {}

    # Loop over fields in dictionary and if they are desired, add them to the
    # dictionary which will be returned:
    field: str
    for field in dict_to_cut:
        if field in fields_to_use:
            cut_dict_for_return[field] = dict_to_cut[field]

    return cut_dict_for_return

def upsert_to_index(data_to_upsert: Any, index_save_path: str) -> None:
    """
    Upserts data to index at given save path. That is, if the index exists then
    new data is appended. If not, the index is created. Allows for indexing in
    batches.
    """
    # Want to generate word embeddings using specified hugging-face model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

    # Index:
    embeddings.upsert(tqdm(data_to_upsert))
    # Save:
    embeddings.save(index_save_path)

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

def save_data(data: Any, data_save_name: str, save_dir: str) -> None:
    '''
    Saves the given python object using pickle. Stores in given directory.
    '''
    # Full path to save data at, including directory and filename:
    data_store_path: str = os.path.join(save_dir, data_save_name)

    # Writing data to file:
    with open(data_store_path, "wb") as file_for_writing_data:
        pickle.dump(data, file_for_writing_data)

def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using pickle.
    '''
    with open(file_path, "rb") as file_to_read:
        return pickle.load(file_to_read)

if __name__ == "__main__":
    sys.exit(main())
