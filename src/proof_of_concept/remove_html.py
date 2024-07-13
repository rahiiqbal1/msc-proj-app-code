import os
import sys
import json
import copy
from bs4 import BeautifulSoup
from get_data_and_embeddings import generate_jsons_from_ndjsons

def main() -> int:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "wikidata"
    )

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

if __name__ == "__main__":
    sys.exit(main())
