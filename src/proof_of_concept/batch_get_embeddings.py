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

    # Get path to wiki-data:
    wiki_data_path: str = os.path.join(
        data_store_path,
        "wikidata",
        "reduced-ndjsons"
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


if __name__ == "__main__":
    main()
