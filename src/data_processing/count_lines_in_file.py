import sys
import os

def main() -> None:
    all_data_dir: str = os.path.join(
        os.pardir, os.pardir, "data"
    )

    file_to_count_path: str = os.path.join(
        all_data_dir, sys.argv[1]
    )

    print("Counting lines...")
    print(count_lines_in_file(file_to_count_path))

    sys.exit(0)

def count_lines_in_file(file_path: str) -> int:
    line_count: int = 0

    with open(file_path, 'r') as file_to_count:
        for _ in file_to_count:
            line_count += 1

    return line_count

if __name__ == "__main__":
    main()
