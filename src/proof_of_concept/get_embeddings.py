import json
import os
import sys
import math
from typing import Any, Generator
from txtai import Embeddings
from tqdm import tqdm
import joblib

PROPORTION_DATA_TO_USE = 0.3

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
        wikidata_dir, "page_titles.gz"
    )
    # Loading in data:
    data_entries: list[str] = gen_or_get_data(
        os.path.join(wikidata_dir, "reduced-ndjsons"), data_save_path
    )

    # Using only num_entries_to_use entries to save computation time:
    num_entries_to_use: int = math.floor(
        len(data_entries) * PROPORTION_DATA_TO_USE
    )
    data_entries_subset: list[str] = data_entries[: num_entries_to_use]

    # If the index does not already exist at the specified path, index and
    # save:
    idx_save_path: str = (
        os.path.join(
            os.path.dirname(wikidata_dir),
            f"embeddings_subset_{num_entries_to_use}"
        )
    )
    # Attempt to generate and save index:
    gen_and_save_index(data_entries_subset, idx_save_path)

    return 0

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
            for json_line in tqdm(ndjson_file, "Loading data"):
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
    ) -> list[str]:
    '''
    If the file at the given path already exists, attempt to load it using
    joblib. If not, read all .ndjson files in the given directory and attempt
    to read name, abstract, categories (todo), and wikitext from each .json 
    within them.

    Returns a list of all fields for a given entry as a single string, e.g.
    ["name... abstract... categories... wikitext...", ...]
    '''
    # If the data does not exist, proceed to generate it. Otherwise load it:
    if os.path.isfile(data_save_path) == False:
        # Initialising list to store strings of data for each entry:
        all_entry_data: list[str] = []

        # List of deserialised json objects per .ndjson file read:
        entry_jsons: list[dict[str, Any]]
        for entry_jsons in generate_jsons_from_ndjsons(ndjson_data_dir):
            # Looping over entry_jsons to get single json objects for each 
            # wikipage entry:
            entry_json: dict[str, Any]
            for entry_json in entry_jsons:
                entry_data: str = ""
                # String to store data of each field for the given entry:
                # Looping over each individual json (dictionary) to add each
                # field to the return string:
                for field in entry_json:
                    entry_data += (entry_json[field] + ' ')

                # Adding string of data to list of data:
                all_entry_data.append(entry_data)

        return all_entry_data

    else:
        # Loads using joblib (pickle), so the file must be valid as the given
        # return type:
        return load_data(data_save_path)
    # if os.path.isfile(data_save_path) == False:
    #     # Getting data to use for indexing:
    #     page_titles: list[str] = []
    #     page_json_list: list[dict[str, Any]]
    #     single_page_json: dict[str, Any]
    #     for page_json_list in generate_jsons_from_ndjsons(ndjson_data_dir):
    #         for single_page_json in page_json_list:
    #             page_titles.append(single_page_json["name"])

    #     # Save parsed titles at specified save path:
    #     title_save_dir, title_save_filename = os.path.split(title_save_path)
    #     save_data(page_titles, title_save_filename, title_save_dir)
        
    #     return page_titles 

    # else:
    #     page_titles: list[str] = load_data(title_save_path)
    #     return page_titles

def gen_and_save_index(
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
