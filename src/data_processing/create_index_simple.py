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

def index_single_json(
    json_to_index: dict[str, str],
    idx_of_json: int # The index of the given json within the collection.
    ) -> dict[str, dict[int, int]]:
    """
    Creates index for a single json object.

    Returns index as dictionary of the form 
    {word: {idx_of_doc: num_of_occurences_of_word, ...}}.
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

def test_index_single_json() -> None:
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

def index_single_ndjson(
    ndjson_file_path: str
) -> dict[str, dict[int, int]]:
    """
    Indexes all json objects in an ndjson. 
    """
    # Initialise list to store indexes for each json in the ndjson:
    list_of_indexes: list[dict[str, dict[int, int]]] = []

    # Loading in jsons:
    list_of_jsons: list[dict[str, str]] = dm.load_jsons_from_ndjson(
        ndjson_file_path
    )

    # Indexing each json and adding the index to the list of them all:
    single_json: dict[str, str]
    for json_idx, single_json in enumerate(list_of_jsons):
        list_of_indexes.append(index_single_json(single_json, json_idx))

    # Combining the indexes and returning:
    return combine_indexes(list_of_indexes)

def combine_indexes(
    indexes_to_combine: list[dict[str, dict[int, int]]]
    ) -> dict[str, dict[int, int]]:
    """
    Combines the indexes in the given list. Requires that there is no crossover
    in document index between the indexes.

    Allowed: {"this": {1: 12}} + {"this": {2: 13}} = {"this": {1: 12, 2: 13}}
    Invalid: {"this": {1: 12}} + {"this": {1: 13}}
    """
    combined_index: dict[str, dict[int, int]] = {}

    single_index: dict[str, dict[int, int]]
    for single_index in indexes_to_combine:
        word: str
        for word in single_index:
            # For each word in the current index, set the value in the combined
            # index to be itself if it already exists, or the value for the
            # current index if not:
            combined_index[word] = combined_index.get(word, single_index[word])
            # Then add the value for the current index to the dictionary for
            # the current word in the combined index:
            combined_index[word].update(single_index[word])

    return combined_index

def test_combine_indexes() -> None:
    print("Testing combine_indexes:\n")

    idx1 = {"this": {1: 12}}
    idx2 = {"this": {2: 12}}

    # 1 + 2:
    print("Expected value: {'this': {1: 12, 2: 12}}")
    print(f"Actual value: {combine_indexes([idx1, idx2])}")
    print("----------")

    idx3 = {"this": {1: 12, 2: 4}, "that": {1: 2, 2: 8}}
    idx4 = {"there": {3: 3, 4: 5}, "that": {3: 2, 4: 6}}

    # 3 + 4:
    print("Expected value: {'this': {1: 12, 2: 4}, 'there': {3: 3, 4: 5}, " +
                           "'that': {1: 2, 2: 8, 3: 2, 4: 6}}")
    print(f"Actual value: {combine_indexes([idx3, idx4])}")
    print("----------")

if __name__ == "__main__":
    # main()
    test_index_single_json()
