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

    # Page title and search box:
    st.title("Wikipedia Search Engine")
    search_query: str = st.text_input("Enter a search query...")

    return 0

# Cache to speed up streamlit load times:
@st.cache_data
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
@st.cache_data
def load_data(file_path: str) -> Any:
    '''
    Loads given object as python object using joblib.
    '''
    return joblib.load(file_path)

if __name__ == "__main__":
    sys.exit(main())
