from typing import List


def load_txt_into_list(filename: str) -> List:
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines


def save_list_into_txt(filename: str, file: List) -> None:
    with open(filename, "w") as output:
        for value in file:
            output.write(str(value) + '\n')
    return None
