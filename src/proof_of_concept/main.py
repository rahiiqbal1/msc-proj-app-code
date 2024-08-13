import os
import sys
import math
import subprocess
from typing import Any
import joblib
from txtai import Embeddings

def main() -> int:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Get data and saved embeddings:
    entry_jsons: list[dict[str, str]]
    embeddings: Embeddings
    entry_jsons, embeddings = load_data_and_embeddings(
        os.path.join(wikidata_dir, "poc_json_data.pkl"),
        os.path.join(wikidata_dir, "poc_embeddings_subset_6947320")
    )

    cli_search(entry_jsons, embeddings)

    return 0
        
def cli_search(data: list[dict[str, str]], embeddings: Embeddings) -> None:
    '''
    Search through json data. uses terminal interface.
    '''
    while True:
        # Clear screen:
        subprocess.run(["clear"])
            
        # Display niceties:
        print("Wikipedia Search")
        print('-' * 50)

        # Getting query and searching through index:
        search_query: str = input("Enter a search query: ")
        # This is a list of tuples containing the index of the result in the
        # data and it's score:
        num_results: list[tuple[int, float]] = embeddings.search(
            search_query, 10
        )

        # Tuple of desired fields of the data to display to the user in
        # results:
        fields_to_show: tuple[str, ...] = ("name", "url")

        # Getting results in readable format:
        num_result: tuple[int, float]
        for num_result in num_results:
            print('+' * 50)
            readable_result: dict[str, str] = data[num_result[0]]
            # Displaying only desired fields:
            field_to_show: str
            for field_to_show in fields_to_show:
                print(readable_result[field_to_show])

        print('-' * 50)
        
        # Break condition:
        to_leave: str = input(
            "Input q/Q to quit, any other key to search again: "
        ).lower()

        if to_leave == 'q':
            break

def test_semantic_search() -> None:
    '''
    Testing whether semantic search works properly for this model: 
    '''
    test_data = [
        "Beans on toast",
        "Capital punishment",
        "Tiger",
        "Generosity"
    ]

    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.index(test_data)

    # print(test_data[embeddings.search("breakfast", 1)[0][0]])

def load_data_and_embeddings(
    data_path: str,
    embeddings_path: str
    ) -> tuple[list[dict[str, str]], Embeddings]:
    '''
    Loads data with joblib (pickle) and it's associated txtai embeddings.

    Returns a tuple of data and embeddings.
    '''
    # Getting data:
    data: list[dict[str, str]] = load_data(data_path)

    # Getting embeddings:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.load(embeddings_path)

    return data, embeddings

def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using joblib.
    '''
    return joblib.load(file_path)

if __name__ == "__main__":
    sys.exit(main())
