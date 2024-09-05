import os
import sys
import subprocess

from txtai import Embeddings

# Bodge:
sys.path.append(os.path.abspath(".."))
from data_processing import data_manipulation as dm

def main() -> None:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data"
    )

    # Path to pickled json data:
    pickled_ndjsons_path: str = os.path.join(
        wikidata_dir, "poc-reduced-ndjsons.pkl"
    )

    # Path to txtai embeddings:
    embeddings_path: str = os.path.join(
        wikidata_dir, "poc-txtai-embeddings-1"
    )

    # Get json data:
    entry_jsons: list[dict[str, str]] = dm.load_data(pickled_ndjsons_path)

    # Get embeddings:
    embeddings = dm.load_embeddings(embeddings_path)

    cli_search(entry_jsons, embeddings)

    sys.exit(0)
        
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

if __name__ == "__main__":
    sys.exit(main())
