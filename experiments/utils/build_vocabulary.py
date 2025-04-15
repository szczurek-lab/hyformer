import re
import argparse

from tqdm import tqdm

from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.auto import AutoDataset


BENCHMARKS = [
    'unimol',
    'guacamol',
    'hi/drd2',
    'hi/hiv',
    'hi/kdr',
    'hi/sol',
    'lo/drd2',
    'lo/kdr',
    'lo/kcnh2',
    'molecule_net/scaffold/bace',
    'molecule_net/scaffold/bbbp',
    'molecule_net/scaffold/clintox',
    'molecule_net/scaffold/esol',
    'molecule_net/scaffold/freesolv',
    'molecule_net/scaffold/hiv',
    'molecule_net/scaffold/lipo',
    'molecule_net/scaffold/sider',
    'molecule_net/scaffold/tox21',
    'molecule_net/scaffold/toxcast'
]

SMILES_REGEX_PATTERN = r"""(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""


class RegexTokenizer:

    def __init__(self, regex_pattern: str = SMILES_REGEX_PATTERN):
        self.regex = re.compile(regex_pattern)

    def get_tokens(self, text):
        return list(set(self.regex.findall(text)))


def main(data_dir, output_filepath):
    
    tokens = []
    tokenizer = RegexTokenizer()
    
    config_filepath = 'configs/datasets/{benchmark}/config.json'

    for benchmark in BENCHMARKS:
        
        config_filepath = config_filepath.format(benchmark=benchmark)
        dataset_config = DatasetConfig.from_config_filepath(config_filepath)
        
        for split in ['train', 'val', 'test']:
            try:
                dataset = AutoDataset.from_config(dataset_config, split=split, root=data_dir)
                for idx in tqdm(range(len(dataset)), desc=f"Extracting tokens from {benchmark} {split} split"):
                    tokens.extend(tokenizer.get_tokens(dataset[idx]['data']))
                tokens = list(set(tokens))  
            except Exception as e:
                print(f"Error loading dataset for {benchmark} {split} split: {e}")
                continue

    tokens.sort()
    
    with open(output_filepath, 'w') as f:
        for token in tokens:
            f.write(token + '\n')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, nargs='?', default=None)
    parser.add_argument('--output_filepath', type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_filepath)
    