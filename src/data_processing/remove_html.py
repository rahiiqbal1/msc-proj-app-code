import os
import sys
import json
import copy
from typing import Any, Generator
from bs4 import BeautifulSoup
from tqdm import tqdm

def main() -> int:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.getcwd()

    clean_html_from_wikitext(
        os.path.join(wikidata_dir, "reduced-ndjsons"),
        os.path.join(wikidata_dir, "reduced-nohtml-ndjsons")
    )

    return 0

def clean_html_from_wikitext(
    ndjson_dir: str,
    processed_ndjson_store_dir: str
    ) -> None:
    '''
    Removes html markup from wikitext field for each json in all ndjsons within
    the given directory.

    Stores processed files in directory specified.
    '''
    # Looping over lists of all jsons per .ndjson file:
    unprocessed_json_list: list[dict[str, str]]
    # Variable to keep track of file number when saving:
    file_num: int = 0
    for unprocessed_json_list in generate_jsons_from_ndjsons(ndjson_dir):
        # Initialising list to store processed json files per .ndjson:
        processed_json_list: list[str] = []

        # Looping over each single json in the current .ndjson file to be able
        # to clean it's wikitext field:
        single_unprocessed_json: dict[str, str]
        for single_unprocessed_json in unprocessed_json_list:
            # Initialising dictionary to store processed json object:
            single_processed_json: dict[str, str] = copy.deepcopy(
                single_unprocessed_json
            )

            # Soupifying, getting text only, and setting text field of copied
            # json dictionary:
            unprocessed_wikitext_soup = BeautifulSoup(
                single_unprocessed_json["wikitext"], "html.parser"
            )
            single_processed_json["wikitext"] = (
                unprocessed_wikitext_soup.get_text()
            )
            # Appending to list of processed jsons for current ndjson as
            # serialised text string:
            processed_json_list.append(
                json.dumps(single_processed_json) + '\n'
            )

        # Saving processed json list as .ndjson file:
        full_processed_ndjson_save_path: str = os.path.join(
            processed_ndjson_store_dir, f"processed_{file_num}"
        )
        with open(full_processed_ndjson_save_path, 'w') as processed_ndjson:
            processed_ndjson.writelines(processed_json_list)

        file_num += 1

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

if __name__ == "__main__":
    sys.exit(main())
