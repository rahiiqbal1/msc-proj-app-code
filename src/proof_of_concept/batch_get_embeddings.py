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

    for test in generate_jsons_from_ndjsons(wiki_data_path):
        pass

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


if __name__ == "__main__":
    main()
