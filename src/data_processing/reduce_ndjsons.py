import os
import json
from typing import Any, Generator
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

import data_manipulation as dm

def main() -> None:
    # Directory in which all data is stored:
    all_data_dir: str = os.path.join(os.pardir, os.pardir, "data")

    # Name of directory in which ndjsons to process are stored:
    full_ndjson_load_dir: str = os.path.join(
        all_data_dir, "raw-ndjsons" 
    )

    # Name of directory in which to store reduced ndjson files:
    reduced_ndjson_save_dir: str = os.path.join(
        all_data_dir, "poc-reduced-ndjsons2"
    )

    # Creating directory to store reduced .ndjson files: 
    try:
        os.mkdir(reduced_ndjson_save_dir)
    except FileExistsError:
        pass

    # Fields which we want to keep within the ndjsons:
    desired_fields: tuple[str, ...] = ("name", "abstract", "url", "wikitext")

    reduce_all_ndjsons(
        full_ndjson_load_dir, reduced_ndjson_save_dir, desired_fields
    )

def reduce_single_json(
    single_json: dict[str, Any],
    desired_fields: tuple[str, ...] | list[str]
    ) -> str | None:
    '''
    Reduces a single JSON as python dictionary to only the chosen fields. 
    Returns reduced JSON as a string.
    '''
    # Initialising dictionary to store reduced json object:
    reduced_json: dict[str, Any] = {}
    field: str
    for field in single_json:
        if field in desired_fields:
            try:
                # If categories, get each category name:
                if field == "categories":
                    cat_info_dict: dict
                    # single_json["categories"] is an array of json objects.
                    # Each of these will be cat_info_dict:
                    for idx, cat_info_dict in enumerate(
                            single_json["categories"]):
                        # Writing "name" subfield to our reduced_json:
                        reduced_json[f"cat{idx}"] = cat_info_dict["name"]

                # If article body, want only wikitext subfield:
                elif field == "article_body":
                    reduced_json["wikitext"] = (
                            single_json["article_body"]["wikitext"])
                
                # Otherwise, store field as-is:
                else:
                    reduced_json[field] = single_json[field]
            except KeyError:
                return None

    # Once all fields have been checked, serialise reduced_json and return:
    return json.dumps(reduced_json)

def reduce_all_jsons_in_ndjson(
    ndjson_file_path: str,
    desired_fields: tuple[str, ...] | list[str]
    ) -> str:
    '''
    Reduces all json objects in a .ndjson file to the desired fields, then
    return a string representation of the reduced ndjson.
    '''
    # Getting list of json dictionaries:
    json_list: list[dict[str, Any]] = dm.load_jsons_from_ndjson(
        ndjson_file_path
    )

    # Getting generator of reduced jsons:
    str_jsons_iterator = map(
        partial(reduce_single_json, desired_fields = desired_fields),
        json_list
    )

    # Getting list of string jsons without None values, then joining on 
    # newline chars and returning. This represents a string .ndjson:
    return "\n".join(
        [str_json for str_json in str_jsons_iterator if str_json is not None]
    )

def reduce_all_ndjsons(
    dir_to_read: str,
    dir_to_store: str,
    desired_fields: tuple[str, ...] | list[str]
    ) -> None:
    '''
    Reduces all .ndjson files in the given directory.
    '''
    # Getting list of file names in given directory:
    file_names: list[str] = os.listdir(dir_to_read)

    # Iterating over each file name and reducing:
    for ndjson_file_name in tqdm(file_names, "Reducing NDJSONs"):
        # Getting full (relative) path of file:
        file_path: str = os.path.join(dir_to_read, ndjson_file_name)
        reduce_all_jsons_in_ndjson(file_path, dir_to_store, desired_fields)
    
    print("NDJSON files in directory successfully reduced.\n" +
          "--------------------")
    
if __name__ == "__main__":
    main()
