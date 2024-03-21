# write a function joins multiple strings into a single path

import os


def get_experiment_dir_path(*args):
    return os.path.join(*convert_to_string(*args))


def convert_to_string(*args):
    return [str(arg) for arg in args]
