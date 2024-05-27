# Jointformer

The official implementation of the [Jointformer](https://arxiv.org/abs/2310.02066), a foundation model for 
small molecule generation and scoring (molecular property prediction). In contrast to existing foundation
models, Jointformer is a [joint model](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf), 
simultaneously exceling in both generation and a prediction, fostering learning meaningful representations, improved calibration
robustness and OOD detection.

## Getting Started

### Installation
To create an environment that satisfies the necessary requirements run
```
 conda env create -f env.yml
```
Next, install Jointformer from the project directory with 
```
conda activate hybrid-transformer
pip install -e .
```

Optionally, for a faster build use [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or
enable [conda-libmamba-solver](https://www.anaconda.com/blog/conda-is-fast-now) with 
``` 
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

### Repository Structure

```
.
├── configs/              # Configuration files
├── data/                 # Datasets and vocabularies
├── experiments/          # Scripts to run experiments
├── results/              # Directory to store results
└── jointformer/          # Source code
    ├── configs/          # Configuration classes
    ├── models/           # Model classes
    ├── trainers/         # Trainer classes
    └── utils/           
        ├── datasets/     # Dataset classes
        ├── tokenizers/   # Tokenizer classes
        └── ...           # Other utility classes

```

## Basic Usage

### Hyperparameters

All hyperparameters, defining tasks, models, trainers and loggers, are stored in `configs/`.

Show running experiment where you have your own data SMILES and properties optionally, maybe actual as a .npy file :) 
Instead of GuacaMol. 

### Vocabularies

Vocabularies are stored in `data/vocabularies/` and can be built with
```python
python experiments/vocabulary/build.py --path_to_task_config <PATH_TO_TASK_CONFIG>
```

### Datasets & Tokenizers

Each task specifies a dataset and tokenizer configuration. As an example, one can download and
load the test split of the unsupervised GuacaMol dataset together with a SMILES tokenizer with

```python
from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer

PATH_TO_TASK_CONFIG = './configs/tasks/guacamol/unsupervised/config.json'

task_config = TaskConfig.from_pretrained(PATH_TO_TASK_CONFIG)

dataset = AutoDataset.from_config(task_config, split='test')
tokenizer = AutoTokenizer.from_config(task_config)

smiles = next(iter(dataset))
inputs = tokenizer(smiles)
```

The tokenizer not only tokenizes the input, but returns all the necessary inputs
for the forward pass of the model i.e. attention masks etc.


### Models

Pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/1t18MULGmZphpjEdPV2FYUYwshEo8W5Dw?usp=sharing)
and initialized with the `AutoModel` class using a model config file. As an example, the following code
loads a pre-trained model and generates a batch of SMILES strings. 

```python
from jointformer.configs.model import ModelConfig
from jointformer.models.auto import AutoModel

PATH_TO_MODEL_CONFIG = './configs/models/jointformer/'
PATH_TO_PRETRAINED_MODEL = './results/pretrain/jointformer/'

model_config = ModelConfig.from_pretrained(PATH_TO_MODEL_CONFIG)
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
from jointformer.configs.trainer import TrainerConfig
from jointformer.trainers.trainer import Trainer

PATH_TO_TRAINER_CONFIG = './configs/trainers/fine-tune/'

trainer_config = TrainerConfig.from_pretrained(PATH_TO_TRAINER_CONFIG)
trainer = Trainer(config=trainer_config, model=model, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```


----
## Experiments

In order to reproduce the experiments, an environment with additional dependencies is required.
To install the necessary dependencies, including [GuacaMol](https://github.com/BenevolentAI/guacamol)
 and [MoleculeNet](https://moleculenet.org/) benchmarks, run
```
conda env create --file jointformer-experiments.yml
```

For installing [MOSES](https://github.com/molecularsets/moses/tree/master), additionally run
```
git clone https://github.com/molecularsets/moses.git
cd moses
python setup.py install
```
and in case Git LFS is not enabled, manually substitute all data files in `moses/data/` and `moses/moses/dataset/data` directories.


### Model Training

To train a model, run 
```
bash experiments/joint_learning/train.sh
```

----
## References


----
## Miscelanous

use TDC as a wrapper.

download ZINC15 data from [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq)

