import os
import sys
from typing import Any
import joblib
from txtai import Embeddings
# import streamlit as st

from get_embeddings import NUM_ENTRIES

def main() -> int:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "wikidata"
    )

    # Get data and saved embeddings:
    embeddings: Embeddings
    entry_data, embeddings = load_data_and_embeddings(
        os.path.join(wikidata_dir, "entry_data.gz"),
        os.path.join(wikidata_dir, f"embeddings_subset_{NUM_ENTRIES}")
    )

    cli_search(entry_data, embeddings)

    return 0

# Cache to speed up streamlit load times:
# @st.cache_data
def load_data_and_embeddings(
    data_path: str,
    embeddings_path: str
    ) -> tuple[list[str], Embeddings]:
    '''
    Loads data with joblib (pickle) and it's associated txtai embeddings.

    Returns a tuple of data and embeddings.
    '''
    # Getting data:
    data: list[str] = load_data(data_path)

    # Getting embeddings:
    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    embeddings.load(embeddings_path)

    return data, embeddings

# Cache to speed up streamlit load times:
# @st.cache_data
def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using joblib.
    '''
    return joblib.load(file_path)

def cli_search(data: list[str], embeddings: Embeddings) -> None:
    '''
    Search through data. uses terminal interface.
    '''
    # Getting query and searching through index:
    search_query: str = input("Enter a search query: ")
    # This is a list of tuples containing the index of the result in the 
    # data and it's score:
    num_results: list[tuple[int, float]] = embeddings.search(search_query, 10)

    # List to store fields for display. Only show a subset of all fields of the
    # data:
    results_to_show: list[str] = []
    # Getting titles of results:
    num_result: tuple[int, float]
    for num_result in num_results:
        # Getting data from index in num_result:
        readable_result: str = data[num_result[0]]
        # Splitting on comma and taking only first element to (roughly) get 
        # only title:
        result_title: str = readable_result.split(',')[0]
        # Adding to list to store data from relevant fields:
        results_to_show.append(result_title)

    # Displaying results:
    for result in results_to_show:
        print(result)

# def streamlit_search(data: list[str], embeddings: Embeddings) -> None:
#     ''' 
#     Searches through given data. Uses a streamlit interface.
#     '''
#     # Page title and search box:
#     st.title("Wikipedia Search Engine")
#     search_query: str = st.text_input("Enter a search query...")

#     # If the search button is pressed, search through the index with the given
#     # query:
#     if st.button("Search") == True and search_query == True:
#         # num_results is a tuple of the index of the given result in the
#         # data and it's score:
#         num_results: list[tuple[int, float]]  = embeddings.search(
#             search_query, 10
#         )

#         # Getting readable results by index of result in page_titles:
#         readable_results: list[str] = [
#             data[num_result[0]] for num_result in num_results
#         ]

#         # Displaying results:
#         for result in readable_results:
#             st.write(result)

if __name__ == "__main__":
    sys.exit(main())
