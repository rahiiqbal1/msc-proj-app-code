import json
import os
import sys
import math
from typing import Any, Generator
from txtai import Embeddings
from tqdm import tqdm
import joblib

NUM_ENTRIES = 6947320
PROPORTION_ENTRIES_TO_USE = 1

def main() -> int:
    # Get path to where data is stored. Note that this uses a relative path
    # from the CWD.
    data_store_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Get path to wiki-data:
    wikidata_dir: str = os.path.join(
        data_store_dir,
        "wikidata"
    )

    # Full path of data file:
    data_save_path: str = os.path.join(
        wikidata_dir, "entry_data.gz"
    )
    # Loading in data:
    data_entries: list[dict[str, str]] = gen_or_get_data(
        os.path.join(wikidata_dir, "reduced-ndjsons"), data_save_path
    )

    # Using only num_entries_to_use entries to save computation time:
    num_entries_to_use: int = math.floor(
        NUM_ENTRIES * PROPORTION_ENTRIES_TO_USE
    )
    data_entries_subset: list[dict[str, str]] = data_entries[
        : num_entries_to_use
    ]

    data_entries_subset_as_strings: list[str] = stringify_dictionaries(
        data_entries_subset
    )

    # If the index does not already exist at the specified path, index and
    # save:
    idx_save_path: str = (
        os.path.join(
            wikidata_dir,
            f"embeddings_subset_{num_entries_to_use}"
        )
    )
    # Attempt to generate and save index:
    gen_or_get_index(data_entries_subset_as_strings, idx_save_path)

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

def gen_or_get_data(
    ndjson_data_dir: str,
    data_save_path: str
    ) -> list[dict[str, str]]:
    '''
    If the file at the given path already exists, attempt to load it using
    joblib. If not, read all .ndjson files in the given directory and attempt
    to read name, abstract, categories, url, and wikitext from each .json 
    within them.

    Returns a list of all .json entries in the data as dictionaries.
    '''
    # List of fields to use for the data:
    fields_to_use: list[str] = ["name", "abstract", "Category", "url"]

    # If the data does not exist, proceed to generate it. Otherwise load it:
    if os.path.isfile(data_save_path) == False:
        # Initialising list to store dictionaries with desired fields:
        all_cut_entry_jsons: list[dict[str, str]] = []

        # List of deserialised json objects per .ndjson file read:
        entry_jsons: list[dict[str, Any]]
        for entry_jsons in generate_jsons_from_ndjsons(ndjson_data_dir):
            # Looping over entry_jsons to get single json objects for each 
            # wikipage entry:
            entry_json: dict[str, Any]
            for entry_json in entry_jsons:
                # Initialising dictionary which will be added to return list:
                cut_json_for_return: dict[str, str] = {}
                # Loop over fields in current json and if they are desired, add
                # them to the dictionary which will be part of the return list:
                for field in entry_json:
                    if field in fields_to_use:
                        cut_json_for_return[field] = entry_json[field]

                # Adding json with selected fields to return list:
                all_cut_entry_jsons.append(cut_json_for_return)

        # Saving data:
        data_save_dir: str
        data_save_name: str
        data_save_dir, data_save_name = os.path.split(data_save_path)
        save_data(all_cut_entry_jsons, data_save_name, data_save_dir)   
        return all_cut_entry_jsons

    else:
        # Loads using joblib, so the file must be valid as the given
        # return type:
        return load_data(data_save_path)

def gen_or_get_index(
    data_to_index: Any,
    index_save_path: str
    ) -> None:
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

def save_data(
    data: Any,
    data_save_name: str,
    save_dir: str,
    compression_level: int = 3
    ) -> list[str]:
    '''
    Saves the given python object using joblib. Stores in given directory.

    If compression_level is given, a supported file extension (.z, .gz, .bz2,
    .xz, .lzma) must be given at the end of data_save_name.
    '''
    # Full path to save data at, including directory and filename:
    data_store_path: str = os.path.join(save_dir, data_save_name)

    return joblib.dump(
        data,
        data_store_path,
        compress = compression_level
    )

def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using joblib.
    '''
    return joblib.load(file_path)

if __name__ == "__main__":
    sys.exit(main())
