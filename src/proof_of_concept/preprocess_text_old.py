import os
import sys
import json
import string
from typing import Any, Generator
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Directory where .ndjson files are stored:
NDJSON_DIR: str = os.path.join(
    os.pardir,
    os.pardir,
    "data",
    "wikidata",
    "reduced-ndjsons"
)

# Directory in which to store processed .ndjsons:
PROCESSED_NDJSON_STORE_DIR: str = os.path.join(
    os.pardir,
    os.pardir,
    "data",
    "wikidata",
    "processed-ndjsons"
)

# List of fields whose text we want to process:
FIELDS_TO_PROCESS: list[str] = [
    "name",
    "abstract",
    "wikitext"
]

# Path to .txt file listing which .ndjson files have already been processed:
NDJSONS_ALREADY_PROCESSED_TXT_PATH = os.path.join(
    os.pardir,
    os.pardir,
    "data",
    "wikidata",
    "ndjsons_already_processed.txt"
)

def main() -> int:
    # Generating deserialised json objects per .ndjson file in the specified
    # directory:
    loaded_json_list: list[dict[str, Any]]
    for idx, loaded_json_list in enumerate(
            generate_jsons_from_ndjsons(
                NDJSON_DIR,
            )
    ):
        # Looping through each json in the list which we loaded in to process
        # it's text:
        processed_ndjson_store_path: str = os.path.join(
            PROCESSED_NDJSON_STORE_DIR, f"processed_ndjson_{idx}"
        )
        # If a file with the name of the path for the current index exists,
        # skip to the next .ndjson. This is so that we can stop and start the
        # processing at different times as it can take a while:
        if os.path.isfile(processed_ndjson_store_path):
            continue

        # An else clause is not needed because the continue will skip this code
        # when desired:
        with open(processed_ndjson_store_path, 'a') as processed_ndjson:
            single_json: dict[str, Any]
            for single_json in tqdm(loaded_json_list, "Processing .jsons"):
                # Write processed json to processing file for current .ndjson
                # file:
                single_json_as_string: str = json.dumps(
                    process_json_text(single_json, FIELDS_TO_PROCESS)
                )
                processed_ndjson.write(single_json_as_string + '\n')

    return 0

def process_json_text(
    json_to_process: dict[str, Any],
    fields_to_process: list[str]
    ) -> dict[str, Any]:
    '''
    Preprocesses text for given fields in given .json file.

    Removes punctuation, stopwords, and lemmatises. Changes to lower case with
    a few exceptions.
    '''
    # Initialising processed json object:
    processed_json: dict[str, Any] = {}

    field: str
    for field in json_to_process:
        if field in fields_to_process:
            # Remove all non-alphabetical characters and convert to lower case:
            processed_json[field] = json_to_process[field].translate(
                str.maketrans('', '', string.punctuation)
            ).lower()
            # Word tokenising to enable lemmatisation and stopword removal:
            word_tokens: list[str] = nltk.word_tokenize(processed_json[field])
            # Lemmatise and remove stopwords:
            lemmatiser = WordNetLemmatizer()
            # Initialising empty string to store lemmatised and stopword-free
            # text:
            lemmatised_nostopword_text: str = ""
            word_token: str
            for word_token in word_tokens:
                if word_token not in stopwords.words("english"):
                    # If the current token is not in the list of stopwords,
                    # lemmatise it and add it to our processed string:
                    lemmatised_nostopword_text += (
                        lemmatiser.lemmatize(word_token) + ' '
                    )
            # Set the processed field text equal to the lemmatised and 
            # stopword-free version of the already punctuation-free and lower
            # case text:
            processed_json[field] = lemmatised_nostopword_text

    return processed_json

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
    ndjsons_dir: str,
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
