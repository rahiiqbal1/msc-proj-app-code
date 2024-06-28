import json
import os
from typing import Any
from txtai import Embeddings
from tqdm import tqdm
import joblib

def main() -> None:
    # Get path to where data is stored. Note that this uses a relative path
    # from the CWD.
    data_store_path: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Get path to sample data:
    wiki_data_path: str = os.path.join(
        data_store_path,
        "wikidata",
        "reduced-ndjsons"
    )
    
    # If the jsons have not already been parsed and saved, do so:
    wiki_jsons_list_path: str = os.path.join(
        os.path.dirname(wiki_data_path), "wiki_jsons_list.tar.gz"
    )
    if not os.path.isfile(wiki_jsons_list_path):
        # Loading in json data into a list of dictionaries. The data in this
        # sample has already been reduced from the full data to a more useable
        # format:
        wiki_jsons: list[dict[str, Any]] = load_all_jsons_from_ndjsons(
            wiki_data_path, 30
        )
        save_data(
            wiki_jsons,
            "wiki_jsons.bz2",
            os.path.dirname(wiki_data_path)
        )
    # Otherwise the file must already exist, so attempt to load it:
    else:
        wiki_jsons: list[dict[str, Any]] = load_data(
            os.path.join(wiki_jsons_list_path)
        )

    # Getting titles to use for indexing: 
    page_titles: list[str] = []
    page_json: dict[str, Any]
    for page_json in wiki_jsons:
        page_titles.append(page_json["name"])
    
    # Generate word embeddings using specified hugging-face model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

    print(page_titles)

    # Index page titles and save index:
    embeddings_path: str = os.path.join(
        data_store_path, "wikidata", "embeddings.tar.gz"
    )
    if not os.path.isfile(embeddings_path):
        embeddings.index(page_titles)
        embeddings.save(embeddings_path)
    else:
        embeddings.load(embeddings_path)

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

def load_all_jsons_from_ndjsons(
    ndjsons_dir: str,
    num_to_read: int
    ) -> list[dict[str, Any]]:
    '''
    Loads in all json objects from a directory of .ndjson files. Stores as 
    python dictionaries in a python list.
    '''
    # Get all filenames in ndjsons_dir to read through them:
    ndjson_filenames: list[str] = os.listdir(ndjsons_dir)

    # Initialising list to store all deserialised jsons:
    json_list_all: list[dict[str, Any]] = []

    num_read: int = 0
    ndjson_filename: str
    for ndjson_filename in tqdm(ndjson_filenames, "Reading .ndjsons"):
        # Getting full filepath as ndjson_filename is only the filename:
        ndjson_filepath: str = os.path.join(ndjsons_dir, ndjson_filename)
        # Adding resulting list to list of all jsons:
        json_list_all += load_jsons_from_ndjson(ndjson_filepath)
        num_read += 1

    return json_list_all

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
    main()
