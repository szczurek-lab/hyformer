import json
import torch

def save_floats_to_file(floats, filename):
    with open(filename, 'w') as f:
        for num in floats:
            f.write(str(num) + '\n')


def read_floats_from_file(filename, dtype='pt'):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    if dtype == 'pt':
        data = torch.Tensor(data)
    return data
