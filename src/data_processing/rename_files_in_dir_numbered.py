import sys
import os

def main() -> None:
    ndjsons_dir: str = os.path.join(
        os.pardir, os.pardir, "data", "poc-reduced-ndjsons"
    )

    rename_files_in_dir_numbered(ndjsons_dir, "reduced")

    sys.exit(0)

def rename_files_in_dir_numbered(directory: str, naming_scheme: str) -> None:
    """
    Renames files in the given directory numerically, according to the given
    naming scheme. 

    e.g. the scheme this_file.txt gives files like this_file_0.txt, 
    this_file_1.txt, etc.
    """
    # Getting filenames in directory. Note that this leads to files being read
    # in a different order than they are represented in:
    filenames_in_dir: list[str] = os.listdir(directory)

    old_filename: str
    for i, old_filename in enumerate(filenames_in_dir):
        old_full_filepath: str = os.path.join(directory, old_filename)

        # New filename for the current file:
        new_filename: str = (
            f"{naming_scheme}_{i}{os.path.splitext(old_filename)[1]}"
        )

        new_full_filepath: str = os.path.join(directory, new_filename)

        os.rename(old_full_filepath, new_full_filepath)

if __name__ == "__main__":
    main()
