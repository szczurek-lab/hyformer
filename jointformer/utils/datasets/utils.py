def save_strings_to_file(strings, filename):
    with open(filename, 'w') as f:
        for s in strings:
            f.write(s + '\n')


def read_strings_from_file(filename):
    with open(filename, 'r') as f:
        strings = f.read().splitlines()
    return strings
