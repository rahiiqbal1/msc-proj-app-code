import os
import sys
from typing import Any
import joblib
from txtai import Embeddings
import streamlit as st

def main() -> int:
    # Directory where relevant data is stored:
    wikidata_dir: str = os.path.join(
        os.pardir,
        os.pardir,
        "data",
        "wikidata"
    )

    # Get title data and saved embeddings. Cache in streamlit for faster update
    # times:
    page_titles: list[str]
    embeddings: Embeddings
    page_titles, embeddings = load_data_and_embeddings(
        os.path.join(wikidata_dir, "page_titles.gz"),
        os.path.join(wikidata_dir, "embeddings_subset_3472510")
    )

    cli_search(page_titles, embeddings)


    return 0

# Cache to speed up streamlit load times:
# @st.cache_data
def load_data_and_embeddings(
    data_path: str,
    embeddings_path: str
    ) -> tuple[list[str], Embeddings]:
    '''
    Loads pickled (with joblib) data and it's associated txtai embeddings.

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

    # Getting readable results from index obtained:
    readable_results: list[str] = [
        data[num_result[0]] for num_result in num_results
    ]

    # Displaying results:
    for result in readable_results:
        print(result)

def streamlit_search(data: list[str], embeddings: Embeddings) -> None:
    ''' 
    Searches through given data. Uses a streamlit interface.
    '''
    # Page title and search box:
    st.title("Wikipedia Search Engine")
    search_query: str = st.text_input("Enter a search query...")

    # If the search button is pressed, search through the index with the given
    # query:
    if st.button("Search") == True and search_query == True:
        # num_results is a tuple of the index of the given result in the
        # data and it's score:
        num_results: list[tuple[int, float]]  = embeddings.search(
            search_query, 10
        )

        # Getting readable results by index of result in page_titles:
        readable_results: list[str] = [
            data[num_result[0]] for num_result in num_results
        ]

        # Displaying results:
        for result in readable_results:
            st.write(result)

if __name__ == "__main__":
    sys.exit(main())
