import os
import sys

from tqdm import tqdm

def main() -> None:
    # Directories to load and save processed data:
    ndjsons_dir: str = os.path.join(
        os.pardir, os.pardir, "data", "poc-reduced-ndjsons"
    )
    combined_ndjson_save_path: str = os.path.join(
        os.pardir, os.pardir, "data", "poc_combined_processed.ndjson"
    )

    combine_ndjsons_in_dir(ndjsons_dir, combined_ndjson_save_path)

    sys.exit(0)

def combine_ndjsons_in_dir(ndjsons_dir: str, final_save_path: str) -> None:
    """
    Combines all .ndjson files in a given directory into a single .ndjson.
    """
    # Get all filenames in ndjsons_dir to read through them:
    ndjson_filenames: list[str] = os.listdir(ndjsons_dir)

    # Opening a file to write the json data to:
    with open(final_save_path, 'a') as ndjson_to_write_to:
        ndjson_filename: str
        for ndjson_filename in tqdm(ndjson_filenames):
            # Getting full filepath as ndjson_filename is only the filename:
            ndjson_filepath: str = os.path.join(ndjsons_dir, ndjson_filename)

            # Reading one json at a time from the current ndjson and adding the
            # line to the final ndjson:
            with open(ndjson_filepath, 'r') as ndjson_to_read:
                ndjson_to_write_to.writelines(ndjson_to_read.readlines())

if __name__ == "__main__":
    main()
