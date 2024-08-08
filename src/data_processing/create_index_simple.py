import sys
import os
import json

from nltk.tokenize import word_tokenize

import data_manipulation as dm

def main() -> None:
    # Directory in which the .ndjson files to be indexed are stored. Here we
    # use the files which have already had their unrelated fields and html 
    # removed, as well as their main body being preprocessed (lower case,
    # stopwords removed, lemmatised, punctuation removed):
    ndjson_store_dir: str = os.path.join(os.pardir, "fully-processed-ndjsons")

    sys.exit(0)

def test() -> None:
    test_json_path: str = os.path.join(
        os.pardir, os.pardir, "data", "testing.json"
    )

    with open(test_json_path, 'r') as json_to_read:
        test_json: dict[str, str] = json.loads(json_to_read.readline())

        test_index: dict[str, list[int]] = index_single_json(test_json, 0)

        print(test_index)

    sys.exit(0)

def index_single_json(
    json_to_index: dict[str, str],
    idx_of_json: int # The index of the given json within the collection.
    ) -> dict[str, list[int]]:
    """
    Creates index for a single json object.

    Returns index as dictionary of the form 
    {word: [num of times word appears in doc at this idx, ...]}.
    """
    # Initialising index:
    index: dict[str, list[int]] = {}

    # Transforming fields of json into a single string for processing:
    json_as_string: str = dm.stringify_dictionaries([json_to_index])[0]

    # Tokenising:
    tokenised_json_text: list[str] = word_tokenize(json_as_string)

    # Looping through and tokenising:
    word: str
    for word in tokenised_json_text:
        try:
            index[word][idx_of_json] += 1
        except IndexError:
            index[word][idx_of_json] = 0

    return index

if __name__ == "__main__":
    # main()
    test()
