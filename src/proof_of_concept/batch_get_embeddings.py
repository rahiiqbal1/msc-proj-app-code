import json
import os
from typing import Any, Generator
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

    # Get path to wiki-data:
    wiki_data_path: str = os.path.join(
        data_store_path,
        "wikidata",
        "reduced-ndjsons"
    )

    # Only get titles if they have not already been saved:
    title_save_path: str = os.path.join(
        os.path.dirname(wiki_data_path), "page_titles.tar.gz"
    )
    if os.path.isfile(title_save_path) == False:
        # Getting titles to use for indexing:
        page_titles: list[str] = []
        page_json_list: list[dict[str, Any]]
        single_page_json: dict[str, Any]
        for page_json_list in generate_jsons_from_ndjsons(wiki_data_path):
            for single_page_json in page_json_list:
                page_titles.append(single_page_json["name"])

        save_data(
            page_titles,
            "page_titles.tar.gz",
            os.path.dirname(title_save_path)
        )

    # Otherwise, get the titles and save them:
    else:
        page_titles: list[str] = load_data(title_save_path)

    # Want to generate word embeddings using specified hugging-face model:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )

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

    # Initialising list to store all deserialised jsons:
    json_list_all: list[dict[str, Any]] = []

    ndjson_filename: str
    for ndjson_filename in tqdm(ndjson_filenames, "Reading .ndjsons"):
        # Getting full filepath as ndjson_filename is only the filename:
        ndjson_filepath: str = os.path.join(ndjsons_dir, ndjson_filename)

        # Yielding list of json dictionaries from the current file:
        yield load_jsons_from_ndjson(ndjson_filepath)

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

if __name__ == "__main__":
    main()
