import sys
import os
import json
import time

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

        start_time: float = time.time()
        test_index: dict[str, dict[int, int]] = index_single_json(test_json, 0)
        end_time: float = time.time()

        print(test_index, '\n', end_time - start_time)

    sys.exit(0)

def index_single_json(
    json_to_index: dict[str, str],
    idx_of_json: int # The index of the given json within the collection.
    ) -> dict[str, dict[int, int]]:
    """
    Creates index for a single json object.

    Returns index as dictionary of the form 
    {word: [num of times word appears in doc at this idx, ...]}.
    """
    # Initialising index:
    index: dict[str, dict[int, int]] = {}

    # Transforming fields of json into a single string for processing:
    json_as_string: str = dm.stringify_dictionaries([json_to_index])[0]

    # Tokenising:
    tokenised_json_text: list[str] = word_tokenize(json_as_string)

    # Looping through and tokenising:
    word: str
    for word in tokenised_json_text:
        # If there is not a value for the word in the index yet, set it's
        # value to a dictionary as below:
        index[word] = index.get(word, {idx_of_json: 0})

        # Add one to the number of occurences of the word in the current 
        # document:
        index[word][idx_of_json] = (index[word][idx_of_json] + 1)

    return index

if __name__ == "__main__":
    # main()
    test()
