# Hyformer

The official implementation of [Hyformer](https://arxiv.org/abs/2310.02066), a [joint model](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf) that simultaneously generates new molecules and predicts their properties.


## Getting Started

### Installation
To create an environment that satisfies the necessary requirements run
```
 conda env create -f env.yml
 conda activate hyformer
```

Optionally, for a faster build use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Installation (reproducing experiments)
To reproduce the experiments from the paper, use
```
conda env create -f env-experiments.yml
conda activate hyformer-experiments
migrate_guacamol.sh
```


## Basic Usage

#### Load a dataset

To load a custom dataset, prepare an `.npz` file that contains an array with the molecular representations and a separate array with their properties. Next, load the dataset using a config file
```
from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.auto import AutoDataset

DATA_DIR = <root_dir/data>

dataset_config = DatasetConfig.from_config_path(DATASET_CONFIG_PATH)
train_dataset = AutoDataset.from_config(dataset_config, split="train", root=DATA_DIR)

_idx = 0
train_dataset[_idx]
```








### Datasets & Tokenizers

Each task specifies a dataset and tokenizer configuration. As an example, one can download and
load the test split of the unsupervised GuacaMol dataset together with a SMILES tokenizer with

```python
from hyformer.configs.task import TaskConfig
from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer

PATH_TO_TASK_CONFIG = './configs/tasks/guacamol/unsupervised/config.json'

task_config = TaskConfig.from_config_path(PATH_TO_TASK_CONFIG)

dataset = AutoDataset.from_config(task_config, split='test')
tokenizer = AutoTokenizer.from_config(task_config)

# Get a sample from the dataset (returns a dictionary with 'data' and 'target' keys)
sample = dataset[0]
smiles = sample['data']
# Note: sample['target'] may be None if no target exists
inputs = tokenizer(smiles)
```

The tokenizer not only tokenizes the input, but returns all the necessary inputs
for the forward pass of the model i.e. attention masks etc.


### Models

Pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1t18MULGmZphpjEdPV2FYUYwshEo8W5Dw?usp=sharing)
and initialized with the `AutoModel` class using a model config file. As an example, the following code
loads a pre-trained model and generates a batch of SMILES strings. 

```python
from hyformer.configs.model import ModelConfig
from hyformer.models.auto import AutoModel

PATH_TO_MODEL_CONFIG = './configs/models/hyformer/'
PATH_TO_PRETRAINED_MODEL = './results/pretrain/hyformer/'

model_config = ModelConfig.from_config_path(PATH_TO_MODEL_CONFIG)
model = AutoModel.from_config(model_config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_pretrained(PATH_TO_PRETRAINED_MODEL)
model.eval()
model.to(device)
model = torch.compile(model)

with torch.no_grad():
    samples = model.generate(
        bos_token_id = '[CLS]',
        eos_token_id = '[SEP]',
        pad_token_id = '[PAD]',
        input_length = 128,
        batch_size = 8,
        temperature=1.0,
        top_k=None,
        device=device
    )
```

Additionally, one can evaluate the perplexity of selected molecule using the dataset and tokenizer
from the example

```python
with torch.no_grad:
    perplexity = model.get_perplexity(**inputs)
```

### Trainers (under construction)

Trainers are used to handle models. A recommended way to initialize the model is with a trainer, initialized using the `AutoTrainer` class and an
appropriate config file. 

```python
from hyformer.configs.trainer import TrainerConfig
from hyformer.trainers.trainer import Trainer

PATH_TO_TRAINER_CONFIG = './configs/trainers/fine-tune/'

trainer_config = TrainerConfig.from_config_path(PATH_TO_TRAINER_CONFIG)
trainer = Trainer(config=trainer_config, model=model, train_dataset=dataset, tokenizer=tokenizer, seed=42)
trainer.train()
```


----
## Experiments

In order to reproduce the experiments, an environment with additional dependencies is required.
To install the necessary dependencies, including [GuacaMol](https://github.com/BenevolentAI/guacamol)
 and [MoleculeNet](https://moleculenet.org/) benchmarks, run
```
conda env create --file hyformer-experiments.yml
```

For installing [MOSES](https://github.com/molecularsets/moses/tree/master), additionally run
```
git clone https://github.com/molecularsets/moses.git
cd moses
python setup.py install
```
and in case Git LFS is not enabled, manually substitute all data files in `moses/data/` and `moses/moses/dataset/data` directories.


### Train

To train a model, run 
```
bash experiments/joint_learning/train.sh
```

### Generate

To train a model, run 
```
bash experiments/joint_learning/train.sh
```

### Evaluate

To train a model, run 
```
bash experiments/joint_learning/train.sh
```

----
## Training Hyformer on new data

To train Hyformer on new data modify the `configs/datasets/sequence/config.json` config by specifying the relative paths to the train/val/test splits of the data.
The data should consist of a data file containing sequences and a property file containing property values of sequences. 

----
## Extending Hyformer to new datasets and tokenizers

In order to train Hyformer to a new dataset, add a 

### Repository Structure

```
.
├── configs/              # configuration files
├── experiments/          # scripts and examples
└── hyformer/          # source code
    ├── configs/            # configurations
    ├── models/             # models
    ├── trainers/           # trainers
    └── utils/           
        ├── datasets/       # datasets
        ├── tokenizers/     # tokenizers
        └── ...             # utilities

```
----
## References


## License 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.0
as published by the Free Software Foundation.
