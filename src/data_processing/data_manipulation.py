"""
Functions here deal with manipulating JSON data, saving, loading it etc, 
aswell as dictionaries in general in some cases.
"""
import os
import sys   
import json
import pickle
import copy
from typing import Any, Generator

from tqdm import tqdm
from txtai import Embeddings

NUMBER_OF_NDJSONS = 372
NUM_ENTRIES = 6947320

def main() -> None:
    all_data_dir: str = os.path.join(os.pardir, os.pardir, "data")

    ndjsons_dir: str = os.path.join(
        all_data_dir, "poc-fully-processed-ndjsons"
    )

    sys.exit(0)

def save_data(data: Any, data_save_name: str, save_dir: str) -> None:
    '''
    Saves the given python object using pickle. Stores in given directory.
    '''
    # Full path to save data at, including directory and filename:
    data_store_path: str = os.path.join(save_dir, data_save_name)

    # Writing data to file:
    with open(data_store_path, "wb") as file_for_writing_data:
        pickle.dump(data, file_for_writing_data)

def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using pickle.
    '''
    with open(file_path, "rb") as file_to_read:
        return pickle.load(file_to_read)

def load_embeddings(embeddings_path: str, mmap = True) -> Embeddings:
    """
    Loads in the txtai embeddings at the given path.
    """
    # Getting embeddings:
    embeddings = Embeddings(
        path = "sentence-transformers/all-MiniLM-L6-v2",
    )
    embeddings.load(embeddings_path, config = {"mmap": mmap})

    return embeddings

def stringify_dictionary(dict_to_stringify: dict[str, str]) -> str:
    """
    Returns a string combining all fields in the given dictionary. 
    """
    return "".join(
        [dict_to_stringify[dict_key] for dict_key in dict_to_stringify]
    )

def stringify_dictionaries(
    dicts_to_stringify: list[dict[str, str]]
    ) -> list[str]:
    '''
    Takes a list of dictionaries which has both keys and values as string data,
    and returns a list of each field in the dictionary as a single combined
    string.
    '''
    return [
        stringify_dictionary(single_dict) for single_dict in dicts_to_stringify
    ]

def _cut_single_dict(
    dict_to_cut: dict[str, Any],
    fields_to_use: tuple[str, ...]
    ) -> dict[str, Any]:
    """
    Selects only the chosen fields from the given dictionary and returns the
    reduced version.

    Returned dictionary must have both keys and values as strings.
    """
    # Copying dict_to_cut to prevent unwanted mutation:
    dict_to_cut_copy = copy.deepcopy(dict_to_cut)

    cut_dict_for_return: dict[str, str] = {}

    # Loop over fields in dictionary and if they are desired, add them to the
    # dictionary which will be returned:
    field: str
    for field in dict_to_cut_copy:
        if field in fields_to_use:
            cut_dict_for_return[field] = dict_to_cut_copy[field]

    return cut_dict_for_return

def cut_list_of_dicts(
    list_of_dicts_to_cut: list[dict[str, Any]],
    fields_to_use: tuple[str, ...]
    ) -> list[dict[str, str]]:
    """
    Cuts all dictionaries in a given list.

    Returns a list of dictionaries with only the chosen fields of the 
    dictionaries from the given list. Chosen fields must have string type 
    values.
    """
    # initialising list to store cut dictionaries:
    list_of_cut_dicts: list[dict[str, str]] = []

    single_dict: dict[str, Any]
    for single_dict in list_of_dicts_to_cut:
        single_cut_dict: dict[str, str] = _cut_single_dict(
            single_dict, fields_to_use
        )

        list_of_cut_dicts.append(single_cut_dict)

    return list_of_cut_dicts

def _save_ndjsons_as_single_pickle(
    ndjsons_dir: str,
    pickle_save_name: str,
    pickle_save_dir: str
    ) -> None:
    """
    Reads all ndjsons in the given directory and saves their contents as a 
    single pickled list of dictionaries (deserialised json objects).
    """
    # Initialising list to store jsons:
    list_of_jsons: list[dict[str, str]] = []

    batch_of_jsons: list[dict[str, str]]
    for batch_of_jsons in generate_jsons_from_ndjsons(ndjsons_dir):
        list_of_jsons += batch_of_jsons

    save_data(list_of_jsons, pickle_save_name, pickle_save_dir)

def _combine_indexes(
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

def _test_combine_indexes() -> None:
    print("Testing combine_indexes:\n")

    idx1 = {"this": {1: 12}}
    idx2 = {"this": {2: 12}}

    # 1 + 2:
    print("Expected value: {'this': {1: 12, 2: 12}}")
    print(f"Actual value: {_combine_indexes([idx1, idx2])}")
    print("----------")

    idx3 = {"this": {1: 12, 2: 4}, "that": {1: 2, 2: 8}}
    idx4 = {"there": {3: 3, 4: 5}, "that": {3: 2, 4: 6}}

    # 3 + 4:
    print("Expected value: {'this': {1: 12, 2: 4}, 'there': {3: 3, 4: 5}, " +
                           "'that': {1: 2, 2: 8, 3: 2, 4: 6}}")
    print(f"Actual value: {_combine_indexes([idx3, idx4])}")
    print("----------")

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
        for json_line in ndjson_file:
            try:
                json_list.append(json.loads(json_line))
            except json.decoder.JSONDecodeError:
                json_list.append({})

    return json_list

def generate_jsons_from_ndjsons(
    ndjsons_dir: str
    ) -> Generator[list[dict[str, Any]], None, None]:
    '''
    Yields at each step a list of all deserialised json objects from a given
    file.
    '''
    # Get all filenames in ndjsons_dir to read through them, and sorting them
    # numerically (requires exactly one number in the filename of each ndjson):
    ndjson_filenames: list[str] = _sort_filenames_with_numbers(
        os.listdir(ndjsons_dir)
    )

    ndjson_filename: str
    for ndjson_filename in tqdm(ndjson_filenames, "Reading .ndjsons"):
        # Getting full filepath as ndjson_filename is only the filename:
        ndjson_filepath: str = os.path.join(ndjsons_dir, ndjson_filename)

        # Yielding list of json dictionaries from the current file:
        yield load_jsons_from_ndjson(ndjson_filepath)

def load_ndjsons_as_single_list(ndjsons_dir: str) -> list[dict[str, Any]]:
    """
    Loads in all ndjson files in the given directory and return a single list
    containing all their jsons.
    """
    # Initialising list which will store json dictionaries:
    list_of_jsons: list[dict[str, Any]] = []

    # Getting each batch and adding it to the list:
    json_batch: list[dict[str, Any]]
    for json_batch in generate_jsons_from_ndjsons(ndjsons_dir):
        list_of_jsons += json_batch

    return list_of_jsons

def generate_jsons_from_single_ndjson(
    ndjson_path: str,
    desired_batch_size: int = 1024
    ) -> Generator[list[dict[str, str]], None, None]:
    """
    Reads a single .ndjson file and returns lists of it's jsons in batches of
    size batch_size.
    """
    # Initialise list to store batches of jsons:
    json_batch: list[dict[str, str]] = []

    with open(ndjson_path, 'r') as ndjson_to_read:
        json_line: str
        for json_line in ndjson_to_read:
            # If the desired batch size has been reached, yield the batch 
            # and clear the list:
            if len(json_batch) == desired_batch_size:
                yield json_batch
                json_batch.clear()

            # Continue adding jsons to the list:
            try:
                json_batch.append(json.loads(json_line))
            except json.decoder.JSONDecodeError:
                json_batch.append({})

        # If we have left the for loop reading lines and there is still some
        # elements in json_batch, then we have some leftover before the full
        # batch size was reached, so yield these:
        if len(json_batch) != 0:
            yield json_batch

def _test_generate_jsons_from_single_ndjson() -> None:
    test_ndjson_path: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "fully-processed-ndjsons",
        "processed_ndjson_0"
    )

    # Getting number of lines in file:
    ndjson_num_lines: int = 0
    with open(test_ndjson_path, 'r') as ndjson_to_read:
        ndjson_num_lines = len(ndjson_to_read.readlines())

    # Initialising variable to keep track of total number of jsons read:
    num_jsons_read: int = 0

    json_batch: list[dict[str, str]]
    for json_batch in generate_jsons_from_single_ndjson(test_ndjson_path):
        num_jsons_read += len(json_batch)

    print(f"Number of lines in ndjson: {ndjson_num_lines}")
    print(f"Number of lines read: {num_jsons_read}")

def create_pickled_cut_jsons(
    ndjson_data_dir: str,
    pickled_list_of_cut_jsons_save_dir: str,
    fields_to_use: tuple[str, ...]
    ) -> None:
    """
    For each .ndjson file in the given directory, select only the specified
    fields from it's JSONs and pickle them as a list[dict[str, str]].
    """
    # Variable to keep track of which ndjson we are on for numeric file naming:
    current_ndjson: int = 0

    wikiarticle_jsons: list[dict[str, str]]
    for wikiarticle_jsons in generate_jsons_from_ndjsons(ndjson_data_dir):
        # Reduce all dictionaries in the current list to the chosen fields and
        # pickle the data:
        cut_json_list_save_name: str = os.path.join(
            pickled_list_of_cut_jsons_save_dir,
            f"list_of_pickled_jsons_{current_ndjson}.pkl"
        )
        with open(cut_json_list_save_name, "wb") as file_for_writing_to:
            pickle.dump(
                cut_list_of_dicts(wikiarticle_jsons, fields_to_use),
                file_for_writing_to
            )

        current_ndjson += 1

def generate_list_of_jsons_from_pickles(
    pickled_data_dir: str
) -> Generator[list[dict[str, str]], None, None]:
    """
    Yields at each step a list of json objects as python dictionaries. Data 
    must be pickled.

    If yield_as_strings = True, then this function yields each list of jsons
    with all their text converted into a single string.
    """
    # Get filenames to read through:
    pickled_data_filenames: list[str] = os.listdir(pickled_data_dir)

    # Sorting filenames for known indexing:
    pickled_data_filenames = _sort_filenames_with_numbers(
        pickled_data_filenames
    )    

    pickled_data_filename: str
    for pickled_data_filename in tqdm(pickled_data_filenames):
        print(pickled_data_filename)
        # Getting absolute path for current file:
        pickled_data_fullpath: str = os.path.join(
            pickled_data_dir, pickled_data_filename
        )

        # Reading pickle and yielding:
        with open(pickled_data_fullpath, "rb") as file_to_read:
            yield pickle.load(file_to_read)

def _get_num_from_string(str_to_parse: str) -> int:
    """
    Takes a string with at most 1 number within and returns that number as an
    int. 

    e.g. this_21 -> 21, that_5222 -> 5222, there0 -> 0, here -> (empty string).
    """
    # Initialising variable to keep track of the number in the string:
    num_in_str: str = ""

    letter: str
    for letter in str_to_parse:
        if letter.isnumeric():
            num_in_str += letter    

    return int(num_in_str)

def _test_get_num_from_string() -> None:
    strings_to_parse: tuple[str, ...] = ("this1", "that24", "those0", "the100")

    string: str
    for string in strings_to_parse:
        print(_get_num_from_string(string))

def _sort_filenames_with_numbers(filenames: list[str]) -> list[str]:
    """
    Takes a list of filenames with exactly one number within them and sorts the
    list numerically.

    e.g. ["a3", "b5", "c1"] -> ["c1", "a3", "b5"].
    """
    # Copying list of filenames to prevent mutation:
    sorted_filenames: list[str] = filenames.copy()

    sorted_filenames.sort(key = _get_num_from_string)

    return sorted_filenames

def _test_sort_filenames_with_numbers() -> None:
    test_names: list[str] = ["a3", "b5", "c1", "d100", "b10"]

    print(_sort_filenames_with_numbers(test_names))

def _test_get_ndjson_line_counts():
    # Filenames and relevant directories:
    sorted_ndjson_filenames: list[str] = ["test1.ndjson", "test2.ndjson"]
    test_ndjsons_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "testing",
        "get-ndjson-line-counts",
        "ndjsons"
    )
    ndjson_line_counts_pickle_save_dir: str = os.path.dirname(test_ndjsons_dir)

    print("Expected: [2, 1]")
    test_ndjson_line_counts: list[int] = _get_ndjson_line_counts(
        sorted_ndjson_filenames,
        test_ndjsons_dir,
        ndjson_line_counts_pickle_save_dir
    )
    print(f"Actual: {test_ndjson_line_counts}")

def _get_ndjson_line_counts(
    sorted_ndjson_filenames: list[str],
    ndjsons_dir: str, 
    ndjson_line_counts_pickle_save_dir: str
    ) -> list[int]:
    """
    Checks if a file exists named "ndjson_line_counts.pkl" in the given
    directory. If it does then it is loaded and returned. If not, then this 
    function calculates the line count of each .ndjson file in the given 
    directory, saves the list in the given directory, and returns.
    """
    # Filename of (potentially) saved pickle:
    ndjson_line_counts_pickle_filename: str = (
        f"{os.path.basename(ndjsons_dir)}-line-counts.pkl"
    )

    # Full path to pickled file of line counts (list[int]):
    ndjson_line_counts_pickle_fullpath: str = os.path.join(
        ndjson_line_counts_pickle_save_dir, ndjson_line_counts_pickle_filename
    )

    # If the pickle exists at the specified path, load it, otherwise calculate
    # and save:
    if os.path.exists(ndjson_line_counts_pickle_fullpath) == True:
        return load_data(ndjson_line_counts_pickle_fullpath)

    else:
        # Getting lengths of files by line number, i.e. the number of jsons:
        ndjson_line_counts: list[int] = []
        single_ndjson_filename: str
        for single_ndjson_filename in sorted_ndjson_filenames:
            # Getting full filepath as filename is only the name:
            full_single_ndjson_filepath: str = os.path.join(
                ndjsons_dir, single_ndjson_filename
            )

            # Getting length of file and adding it to the list of lengths:
            ndjson_line_counts.append(
                count_lines_in_file(full_single_ndjson_filepath)
            )

        save_data(
            ndjson_line_counts,
            ndjson_line_counts_pickle_filename,
            ndjson_line_counts_pickle_save_dir
        )

        return ndjson_line_counts

def find_json_given_index_ndjsons(
    ndjsons_dir: str,
    index: int,
    ndjson_line_counts_pickle_save_dir: str
    ) -> dict[str, str]:
    """
    For a given directory containing numerically-sorted .ndjson files, and a 
    given index, returns the json object specified by that index, treating all
    .ndjsons as a single corpus.
    """
    # Retrieving filenames within directory and sorting. Requires that each
    # ndjson filename contains exactly one number to sort by:
    sorted_ndjson_filenames: list[str] = _sort_filenames_with_numbers(
        os.listdir(ndjsons_dir)
    )

    # Getting line counts of ndjsons in directory:
    ndjson_line_counts: list[int] = _get_ndjson_line_counts(
        sorted_ndjson_filenames, 
        ndjsons_dir,
        ndjson_line_counts_pickle_save_dir
    )

    # Initialising variable to keep track of which file within the directory
    # we are in the scope of through the for loop. Starting from 0 as we can 
    # use the list of sorted filenames to load the file:
    batch_counter: int = 0
    # Searching through lengths till the index is contained:
    single_ndjson_line_count: int
    for single_ndjson_line_count in ndjson_line_counts:
        if index - single_ndjson_line_count >= 0:
            # Subtracting from the index so that we can simply use this
            # variable to index into the correct file when it is found as this
            # will given the index within that file:
            index -= single_ndjson_line_count
            batch_counter += 1
        else:
            break

    # Full path to file which contains the desired data point: 
    containing_ndjson_path: str = os.path.join(
        ndjsons_dir, sorted_ndjson_filenames[batch_counter]
    )

    # Initialising variable which will be set to the desired json if it is 
    # found:
    desired_json: dict[str, str] = {}

    # Reading ndjson at filepath found and retrieving the json at the required
    # index:
    with open(containing_ndjson_path, 'r') as ndjson_to_read:
        single_json: str
        for i, single_json in enumerate(ndjson_to_read):
            if i == index:
                desired_json = json.loads(single_json)
                break

    return desired_json

def _test_find_jsons_given_index_ndjsons() -> None:
    this_test_dir: str = os.path.join(
        os.pardir, os.pardir, "data", "testing", "find-index-ndjsons-test"
    )
    test_ndjsons_dir = os.path.join(this_test_dir, "ndjsons")

    print("TEST 1")
    print("Expected: {'this': 'hello'}")
    test1_result: dict[str, str] = find_json_given_index_ndjsons(
        test_ndjsons_dir, 0, this_test_dir
    )
    print(f"Actual: {test1_result}")
    print('-' * 30)

    print("TEST 2")
    print("Expected: {'this': 'cool'}")
    test2_result: dict[str, str] = find_json_given_index_ndjsons(
        test_ndjsons_dir, 2, this_test_dir
    )
    print(f"Actual: {test2_result}")
    print('-' * 30)

def find_data_given_index_pickles(data_dir: str, index: int) -> dict[str, str]:
    """
    For a directory containing numerically-sorted pickled lists of data, and a
    given index, return the data point from within the file specified by that
    index.
    """
    # Retrieving filenames within directory and sorting:
    sorted_filenames: list[str] = _sort_filenames_with_numbers(
        os.listdir(data_dir)
    )

    # Getting lengths of files:
    file_list_lengths: list[int] = []
    filename: str
    for filename in sorted_filenames:
        # Getting full filepath as filename is only the name:
        filepath_full: str = os.path.join(data_dir, filename)

        # Loading data and adding it's length to the list of lengths:
        file_list_lengths.append(
            len(load_data(filepath_full))
        )

    # Initialising variable to keep track of which file within the directory
    # we are in the scope of through the for loop. Starting from 0 as we can 
    # use the list of sorted filenames to load the file:
    batch_counter: int = 0
    # Searching through lengths till the index is contained:
    list_length: int
    for list_length in file_list_lengths:
        if index - list_length >= 0:
            # Subtracting from the index so that we can simply use this
            # variable to index into the correct file when it is found as this
            # will given the index within that file:
            index -= list_length
            batch_counter += 1

        else:
            break

    # Full path to file which contains the desired data point: 
    containing_file_path: str = os.path.join(
        data_dir, sorted_filenames[batch_counter]
    )

    # Loading correct data in directory and getting json at required index:
    return load_data(containing_file_path)[index]

def _test_find_data_given_index_pickles() -> None:
    test_pickled_lists_dir: str = os.path.join(
        os.pardir, os.pardir, "data", "test-pickled-list-retrieval"
    )
    test_index: int = 7

    lists_to_pickle: tuple[list[int], ...] = (
        [1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]
    )

    # Pickling data:
    list_to_pickle: list[int]
    for fileIdx, list_to_pickle in enumerate(lists_to_pickle):
        save_data(list_to_pickle, f"{fileIdx}.pkl", test_pickled_lists_dir)

    print(find_data_given_index_pickles(test_pickled_lists_dir, test_index))

def count_lines_in_file(file_path: str) -> int:
    line_count: int = 0

    with open(file_path, 'r') as file_to_count:
        for _ in file_to_count:
            line_count += 1

    return line_count

if __name__ == "__main__":
    main()
