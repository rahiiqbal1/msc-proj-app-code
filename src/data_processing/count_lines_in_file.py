import sys
import os

import data_manipulation as dm

def main() -> None:
    all_data_dir: str = os.path.join(
        os.pardir, os.pardir, "data"
    )

    file_to_count_path: str = os.path.join(
        all_data_dir, sys.argv[1]
    )

    print("Counting lines...")
    print(dm.count_lines_in_file(file_to_count_path))

    sys.exit(0)

if __name__ == "__main__":
    main()
