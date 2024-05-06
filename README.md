# Jointformer

This is the official implementation of the Jointformer. 

Jointformer is a transformer-based joint generative model. 

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

## Basic Usage

### Config Files

All hyperparameters are stored in the `configs` directory. In this way you can easily change
the configuration of the dataset, tokenizer, model and the trainer being used, by
modifying the corresponding config file. We ensure that no hyperparameters are set
in the code implicitly fostering transparent and easy reproducibility of the result and
transparent usage of the repository.

### Data & Tokenization

Datasets are stored in the `data` directory. The `AutoDataset` class can be used to load
any dataset by specifying an appropriate config file. Analogously, tokenizers are stored
in the `jointformer/utils/tokenizers` directory and the `AutoTokenizer` class
can be used to load any tokenizer by specifying an appropriate config file. 

While Datasets should handle downloading the data and preprocessing it automatically, 
Tokenizers do require a vocabulary file containing all tokens used to tokenize an input.
Default vocabularies are stored in the `data/vocabularies` directory and a new vocabulary
can be built by extracting all tokens from a selected dataset with `experiments/vocabulary/build.py` script.
Note that Datasets may additionally augment the input data under the hood. 

As an example, the following code loads the test split of the unsupervised
GuacaMol dataset together with a SMILES tokenizer

```python
from jointformer.configs.task import TaskConfig
from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer

PATH_TO_TASK_CONFIG = './configs/tasks/guacamol/unsupervised/config.json'

task_config = TaskConfig.from_pretrained(PATH_TO_TASK_CONFIG)

dataset = AutoDataset.from_config(task_config, split='test')
tokenizer = AutoTokenizer.from_config(task_config)

example_id = 0
smiles = dataset[example_id]
inputs = tokenizer(smiles)
```

The tokenizer not only tokenizes the input, but also returns all the necessary inputs
for the forward pass of the model.


### Pre-trained Models

Pre-trained models can be downloaded from ... and initialized using the `AutoModel` class, given 
a model config file.

As an example, the following code loads a pre-trained model and 
generates a batch of SMILES strings. 

```python
from jointformer.configs.model import ModelConfig
from jointformer.models.auto import AutoModel

PATH_TO_MODEL_CONFIG = './configs/models/jointformer/'
PATH_TO_PRETRAINED_MODEL = './results/pretrain/jointformer/'

model_config = ModelConfig.from_pretrained(PATH_TO_MODEL_CONFIG)
model = AutoModel.from_config(model_config)
model.load_pretrained(PATH_TO_PRETRAINED_MODEL)
model.eval()
model.to('cuda' if torch.cpu.is_available() else 'cpu')
model = torch.compile(model)

with torch.no_grad():
    outputs = model.generate(**inputs)
```

Additionally, one can evaluate the perplexity of a selected molecule, using the dataset and tokenizer
from the previous example

```python
with torch.no_grad:
    perplexity = model.get_perplexity(**inputs)
```

However, a recommended way is to use a trainer to initialize the model. 

### Trainers & fine-tuning
you can additionally fine-tune the model by running

```python
from jointformer.configs.trainer import TrainerConfig
from jointformer.trainers.trainer import Trainer

PATH_TO_TRAINER_CONFIG = './configs/trainers/fine-tune/'

trainer_config = TrainerConfig.from_pretrained(PATH_TO_TRAINER_CONFIG)
trainer = Trainer(config=trainer_config, model=model, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```

## Experiments

The following code reproduces all experiments from the paper.

### Model Training

To train a model run 
```
python experiments/joint_learning/train.py 
  --out_dir ./results/joint_learning/{MODEL}
  --path_to_model_config ./configs/models/{MODEL}
  --path_to_trained_config ./configs/trainers/joint_learning
```

### Pre-Training 

To pre-train a model run 
```
python scripts/pretrain/train.py 
  --out_dir ./results/pretrain/{MODEL}
  --path_to_model_config ./configs/models/{MODEL}
```
to eval
```
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/pretrain/eval.py --out_dir ./results/pretrain/{MODEL} --path_to_model_config ./configs/models/{MODEL}
```

### Joint Learning

Joint learning takes a pre-trained model and trains it jointly on generation and prediction tasks 

```
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/joint_learning/train.py 
  --out_dir ./results/joint_learning/
```

Evaluate with
```
CUDA_VISIBLE_DEVICES={GPU_ID} python scripts/joint_learning/evaluate.py 
  --out_dir ./results/joint_learning
  --data_reference_file ./data/guacamol/test/smiles.txt
```

```
CUDA_VISIBLE_DEVICES=3 python scripts/joint_learning/eval.py --out_dir /raid/aizd/hybrid_transformer/results/table_1/ --data_reference_file ./data/guacamol/test/smiles.txt
```

## Downloads

Download ZINC15 data from [here](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq)

## TODOs

  - scripts refer to names, which refer to benchmarks by a separate file linking configs to names
  - no other parameters should be passed to scripts or set in scripts manually
  - log console outputs and dump to file all the time 

## Results

## References
