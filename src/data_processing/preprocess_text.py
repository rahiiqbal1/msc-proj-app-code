import sys
import os
import string
import json
from multiprocessing import Pool
from typing import Any

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm

import data_manipulation as dm

def main() -> None:
    # Directory in which the .ndjson files to be processed are stored. Here we
    # use the files which have already had their unrelated fields and html 
    # removed:
    unprocessed_ndjson_store_dir: str = os.path.join(
        os.pardir, "poc-reduced-nohtml-ndjsons"
    )
    # Directory within which to store fully processed .ndjson files:
    processed_ndjson_store_dir: str = os.path.join(
        os.pardir, "poc-fully-processed-ndjsons"
    )

    process_ndjsons(unprocessed_ndjson_store_dir, processed_ndjson_store_dir)

    sys.exit(0)

def process_ndjsons(
    unprocessed_ndjson_dir: str,
    processed_ndjson_dir: str
    ) -> None:
    """
    Processes text for all .ndjson files in given directory. Performs
    lower-casing, stopword removal, lemmatisation, and removal of punctuation.
    Does these in one run for faster processing.
    """
    # Variable to keep track of which file we are on for accurate filenames:
    current_ndjson_num: int = 0

    # Looping over all unprocessed .ndjson files in the given directory:
    jsons_in_ndjson: list[dict[str, Any]]
    for jsons_in_ndjson in dm.generate_jsons_from_ndjsons(
            unprocessed_ndjson_dir
    ):
        # Getting list of processed jsons for the current ndjson as lines of 
        # text:
        processed_json_list: list[str] = process_single_ndjson(
            jsons_in_ndjson
        )

        # Writing to new .ndjson file:
        full_processed_ndjson_save_path: str = os.path.join(
            processed_ndjson_dir,
            f"processed_ndjson_{current_ndjson_num}"
        )
        with open(full_processed_ndjson_save_path, 'w') as processed_ndjson:
            processed_ndjson.writelines(processed_json_list)

        current_ndjson_num += 1

def process_single_ndjson(
    jsons_in_ndjson: list[dict[str, Any]],
    ) -> list[str]:
    """
    Processes given text fields for all jsons within the given list. 

    Returns a list of jsons as strings separated by newlines, for writing to
    .ndjson file.
    """
    # Pool for multiprocessing:
    pool = Pool()

    # Processing jsons in parallel:
    processed_json_list: list[str] = list(tqdm(
        pool.imap(
            process_single_json, jsons_in_ndjson
        ),
        total = len(jsons_in_ndjson)
    ))

    pool.close()
    pool.join()

    return processed_json_list

def process_single_json(
    json_to_process: dict[str, Any],
    ) -> str:
    """
    Processes given text fields in given json object.

    Returns as serialised json string with newline character at it's end.
    """
    fields_to_process: tuple[str, ...] = ("name", "abstract", "wikitext")

    # Creating new json to store processed fields:
    processed_json: dict[str, Any] = {}

    # Looping through fields of json:
    field: str
    for field in json_to_process:
        # Getting text within field:
        field_text: str = json_to_process[field]

        if field in fields_to_process:
            field_text = process_single_string(field_text)

        processed_json[field] = field_text

    return json.dumps(processed_json) + '\n'

def process_single_string(text_to_process: str) -> str:
    """
    Processes a single string. Performs lower casing, removal of punctuation,
    lemmatisation, and stopword removal.
    """
    # Making lower case:
    text_to_process = text_to_process.lower()

    # Removing punctuation:
    text_to_process = text_to_process.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Getting list of words in text:
    words_of_text: list[str] = word_tokenize(text_to_process)

    # Removing stopwords and lemmatising:
    text_words_no_stopwords_lemmatised: list[str] = []
    lemmatiser = WordNetLemmatizer()
    word: str
    for word in words_of_text:
        if word not in stopwords.words("english"):
            text_words_no_stopwords_lemmatised.append(
                lemmatiser.lemmatize(word)
            )

    # Joining back into a single string and returning:
    return ' '.join(text_words_no_stopwords_lemmatised)

if __name__ == "__main__":
    main()
