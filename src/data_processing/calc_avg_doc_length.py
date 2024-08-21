import sys
import os
from statistics import mean

from nltk.tokenize import word_tokenize
from tqdm import tqdm

import data_manipulation as dm

def main() -> None:
    # Directory in which data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir, os.pardir, "data"
    )
    # Path to ndjson we wish to check:
    ndjson_file_path: str = os.path.join(
        wikidata_dir, "poc_combined_processed.ndjson"
    )

    # Calculating average document length:
    print(calc_avg_doc_length_in_ndjson(ndjson_file_path))

    sys.exit(0)

def calc_avg_doc_length_in_ndjson(ndjson_file_path: str) -> float:
    """
    Calculates the average length of the json files within the given ndjson.
    """
    # Getting jsons:
    jsons_in_ndjson: list[dict[str, str]] = dm.load_jsons_from_ndjson(
        ndjson_file_path
    )

    # Converting jsons to strings:
    jsons_as_strings: list[str] = dm.stringify_dictionaries(jsons_in_ndjson)

    # Initialising list to store lengths of each json:
    json_lengths: list[int] = []

    # Calculating the length of each json and adding in to the list:
    single_json_str: str
    for single_json_str in tqdm(jsons_as_strings):
        json_lengths.append(calc_single_json_length(single_json_str))

    # Finding average length and returning:
    return mean(json_lengths)

def calc_single_json_length(json_str: str) -> int:
    """
    Calculates the length in words of a single json object, where the json has 
    been converted to a string. 

    Counts the number of words in each field of the json.
    """
    words_in_json: list[str] = word_tokenize(json_str)

    return len(words_in_json)


if __name__ == "__main__":
    main()
