from typing import List


def save_strings_to_file(strings, filename):
    with open(filename, 'w') as f:
        for s in strings:
            f.write(s + '\n')


def read_strings_from_file(filename):
    with open(filename, 'r') as f:
        strings = f.read().splitlines()
    return strings


def load_txt_into_list(filename: str) -> List:
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines


def save_list_into_txt(filename: str, file: List) -> None:
    with open(filename, "w") as output:
        for value in file:
            output.write(str(value) + '\n')
    return None
