from typing import Any
import os
import json
from tqdm import tqdm

def main() -> None:
    # Creating directory to store reduced .ndjson files: 
    try:
        # Total bodge:
        os.mkdir("reduced-ndjsons")
    except FileExistsError:
        pass

    reduce_all_ndjsons("ndjsons", "reduced-ndjsons")

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
            # Uncomment below for status bar:
            # for json_line in tqdm(ndjson_file, "Loading data"):
            for json_line in ndjson_file:
                json_list.append(json.loads(json_line))
        except json.decoder.JSONDecodeError:
            pass

    # print("Data loaded Successfully.\n--------------------")
    return json_list

def reduce_single_json(
    single_json: dict[str, Any],
    list_of_str_jsons: list[str],
    ) -> None:
    '''
    Reduces a single JSON as python dictionary to only the chosen fields. 
    Appends string JSON to given list.
    '''
    # Want name, abstract, category names numerical subfields, and article body
    # wikitext subfield:
    desired_fields: tuple[str, ...] = (
            "name",
            "abstract",
            "categories",
            "article_body",
            "url"
    )

    # Initialising dictionary to store reduced json object:
    reduced_json: dict[str, Any] = {}
    field: str
    for field in single_json:
        if field in desired_fields:

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

    # Once all fields have been checked, serialise reduced_json and add to
    # ndjson_to_write:
    list_of_str_jsons.append(json.dumps(reduced_json))

def reduce_all_jsons_in_ndjson(
    ndjson_file_path: str,
    storage_dir: str
    ) -> None:
    '''
    Reduces all json objects in a .ndjson file to the desired fields.
    '''
    # Getting list of json dictionaries:
    json_list: list[dict[str, Any]] = load_jsons_from_ndjson(ndjson_file_path)
    # Initialising list to store json strings:
    list_of_str_jsons: list[str] = []

    # Getting only desired fields:
    fail_count: int = 0
    # Uncomment below for status bar:
    # for single_json in tqdm(json_list, "Reducing JSONs"):
    for single_json in json_list:
        try:
            reduce_single_json(single_json, list_of_str_jsons)
        except KeyError:
            fail_count += 1

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

def reduce_all_ndjsons(dir_to_read: str, dir_to_store: str) -> None:
    '''
    Reduces all .ndjson files in the given directory.
    '''
    # Getting list of file names in given directory:
    file_names: list[str] = os.listdir(dir_to_read)

    # Iterating over each file name and reducing:
    for ndjson_file_name in tqdm(file_names, "Reducing NDJSONs"):
        # Getting full (relative) path of file:
        file_path: str = os.path.join(dir_to_read, ndjson_file_name)
        reduce_all_jsons_in_ndjson(file_path, dir_to_store)
    
    print("NDJSON files in directory successfully reduced.\n" +
          "--------------------")
    
if __name__ == "__main__":
    main()
