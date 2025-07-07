# Hyformer

The official implementation of [Hyformer](https://arxiv.org/abs/2310.02066), a [joint model](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf) that simultaneously generates new molecules and predicts their properties.

Despite transformers being state-of-the-art models across both molecule generation and property prediction, both tasks are apprached separately, with specialized models developed for each of the tasks. Hyformer unifies these tasks, by effectively combining a transformer-encoder and decoder in a single model. We find that unifying these tasks provides synergistic benefits, including better conditional generation, representation learning and out-of-dictribution property prediction.  

---
## Installation

To install Hyformer via pip directly from GitHub, run
```
pip install git+https://github.com/adamizdebski/hyformer.git@jointformer-2.0#egg=hyformer
```

### Installing with uv

To install Hyformer using [uv](https://docs.astral.sh/uv/), run
```
uv add git+https://github.com/adamizdebski/hyformer.git@jointformer-2.0#egg=hyformer
```

### Installing with conda

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
 

### Reproducing experiments
You can choose between `env_minimal.yml` for a minimal installation, `env.yml` for a minimal installation that includes chemical utils and `env_experiments.yml` for reproducing all experiments from the paper.

To reproduce the experiments from the paper, use
```
conda env create -f env_experiments.yml
conda activate hyformer_experiments
migrate_guacamol.sh
```

## Model Checkpoints

Pre-trained Hyformer models are available on Hugging Face:

### Molecular Models

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| Hyformer-Base | Base model trained on molecular data | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-base) |
| Hyformer-Property | Joint generation and property prediction model | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-property) |
| Hyformer-Large | Large-scale model for enhanced performance | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-large) |

### Peptide Models

| Model | Description | Hugging Face Link |
|-------|-------------|-------------------|
| Hyformer-Peptide-Base | Base model trained on peptide sequences | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-base) |
| Hyformer-Peptide-Property | Joint peptide generation and property prediction | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-property) |
| Hyformer-Peptide-Large | Large-scale peptide model | [🤗 HF Model](https://huggingface.co/adamizdebski/hyformer-peptide-large) |


## Quickstart

To get started, load a pre-trained model from Hugging Face or a local checkpoint:
```python
from transformers import AutoModel, AutoTokenizer

model_name = "adamizdebski/hyformer-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

---

## Basic Usage

#### Load a dataset

To load a custom dataset, prepare an `.npz` file that contains an array with the molecular representations and a separate array with their properties. Next, load the dataset using a config file
```
from hyformer.configs.dataset import DatasetConfig
from hyformer.utils.datasets.auto import AutoDataset

DATA_DIR = <root_dir/data>

dataset_config = DatasetConfig.from_config_filepath(DATASET_CONFIG_PATH)
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

task_config = TaskConfig.from_config_filepath(PATH_TO_TASK_CONFIG)

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

model_config = ModelConfig.from_config_filepath(PATH_TO_MODEL_CONFIG)
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

trainer_config = TrainerConfig.from_config_filepath(PATH_TO_TRAINER_CONFIG)
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


## License 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License v3.0
as published by the Free Software Foundation.

## Citation

Cite our work with 
```
@misc{izdebski2025synergisticbenefits,
      title={Synergistic Benefits of Joint Molecule Generation and Property Prediction}, 
      author={Adam Izdebski and Jan Olszewski and Pankhil Gawade and Krzysztof Koras and Serra Korkmaz and Valentin Rauscher and Jakub M. Tomczak and Ewa Szczurek},
      year={2025},
      eprint={2504.16559},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16559}, 
}
```


