from typing import List


def load_txt_into_list(filename: str) -> List:
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines
