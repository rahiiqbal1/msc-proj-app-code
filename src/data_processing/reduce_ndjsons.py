from typing import Any
import os
import json
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
    storage_dir: str,
    desired_fields: tuple[str, ...] | list[str]
    ) -> None:
    '''
    Reduces all json objects in a .ndjson file to the desired fields.
    '''
    # Getting list of json dictionaries:
    json_list: list[dict[str, Any]] = dm.load_jsons_from_ndjson(
        ndjson_file_path
    )
    # Initialising list to store json strings:
    list_of_str_jsons: list[str] = []

    # Getting only desired fields:
    for single_json in json_list:
        try:
            reduce_single_json(single_json, list_of_str_jsons, desired_fields)
        except KeyError:
            pass

    # Name of .ndjson file to append to full storage path for reduced version:
    ndjson_file_name: str = os.path.basename(ndjson_file_path)
    # Path to store file:
    reduced_ndjson_store_path: str = os.path.join(
        storage_dir, f"reduced_{ndjson_file_name}"
    )
    # Joining string jsons in list_of_str_jsons into single string and writing
    # to file:
    ndjson_to_write: str = "\n".join(list_of_str_jsons)
    with open(reduced_ndjson_store_path, 'a') as reduced_ndjson_file:
        reduced_ndjson_file.write(ndjson_to_write)

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
