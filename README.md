# Hyformer

This repository is the official implementation of [Hyformer](https://arxiv.org/abs/2504.16559), a [joint](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf) transformer-based model that unifies a generative decoder with a predictive encoder. Depending on the task, Hyformer uses either a causal or a bidirectional mask and returns token probabilities or predicted property values.

<img src="hyformer.png" width="520" height="250"/>

> For an optimized implementation, see [Hyformer 2.0](https://github.com/szczurek-lab/hyformer/tree/hyformer-2.0). 


## Installation

To create an environment that satisfies the necessary requirements run
```bash
micromamba create -y -p <ENV_PATH> -f hyformer.yml
bash migrate_guacamol.sh 
```

Verify installation with `python3 scripts/verify_intallation.py`.

## Pre-trained Models

Download pre-trained models from [HuggingFace](https://huggingface.co/SzczurekLab/hyformer):

- [Hyformer-8M](https://huggingface.co/SzczurekLab/hyformer/tree/main/molecules/8M) trained on GuacaMol dataset [1].
- [Hyformer-50M](https://huggingface.co/SzczurekLab/hyformer/tree/main/molecules/50M) trained on 19M molecules from combined: ZINC, ChEMBL and various purchusable molecular datasets [2]. 

### Pre-train from scratch

To pre-train Hyformer from scratch, run

```bash
srun python3 scripts/pretrain/train.py
    --path_to_dataset_config <PATH_TO_DATASET_CONFIG>
    --path_to_tokenizer_config <PATH_TO_TOKENIZER_CONFIG>
    --path_to_model_config <PATH_TO_MODEL_CONFIG>
    --path_to_trainer_config <PATH_TO_TRAINER_CONFIG>
```

## Example Usage

### Featurize

In order to featurize a list of sequences, e.g., SMILES, run

```bash
python scripts/featurize.py \
    --path_to_sequence_file data/raw/sequences.csv \
    --path_to_sequence_column smiles \
    --path_to_output_file data/processed/embeddings.npz \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config models/hyformer/50M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --device cuda:0 \
    --batch_size 256 \
    --seed 1337
```

> Alternatively, `path_to_sequence_file` can point to a `.txt` or `.smiles` file. 

### Predict

To predict target properties, using a fine-tuned model, run
```bash
python3 scripts/predict.py \
    --path_to_sequence_file data/raw/sequences.csv \
    --path_to_sequence_column smiles \
    --path_to_output_file predictions.csv \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config configs/models/hyformer/50M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --device cuda:0 \
    --batch_size 256 \
    --seed 1337
```

### Generate

To unconditionally generate a list of sequences, e.g., SMILES, run
```bash
python3 scripts/generate.py \
    --path_to_output_file data/synthetic/smiles.txt \
    --path_to_tokenizer_config configs/tokenizers/smiles/deepchem/config.json \
    --path_to_model_config configs/models/hyformer/50M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --device cuda:0 \
    --batch_size 16 \
    --seed 1337 \
    --temperature 0.9 \
    --top_k 25 \
    --num_samples 100
```


## Experiments

Experiments are executable through scripts in `experiments/`.

### GuacaMol distribution learning benchmark

To evaluate the unconditional generative performance of Hyformer, using GuacaMol benchmark, run 
```bash
python3 scripts/pretrain/evaluate_guacamol.py \
    --path_to_tokenizer_config configs/tokenizers/smiles/guacamol/config.json \
    --path_to_model_config configs/models/hyformer/8M/config.json \
    --path_to_model_ckpt <PATH_TO_MODEL_CKPT> \
    --path_to_output_file <RESULTS_FILENAME> \
    --device 'cuda:0' \
    --batch_size 256 \
    --temperature 1.0 \
    --top_k 10 \
    --chembl_training_file <PATH_TO_GUACAMOL_TRAINING_FILE>
```

> Guacamol training file can be downloaded [here](https://ndownloader.figshare.com/files/13612760).

> Make sure to first run `migrate_guacamol.sh`. 

### Out-of-Distribution Molecular Property Prediction (Hi benchmark)

```bash
python3 scripts/finetune/run_hi_benchmark.py \
    ...
```


### Conditional molecule generation

For the conditional sampling experiment, first jointly finetune the model
```train
python3 scripts/conditional_sampling/run_surrogate.py \
    ...
```
and generate
```train
python3 scripts/conditional_sampling/generate.py \
    ...
```

## Cite

To cite our work, use

```
@misc{izdebski2025synergisticbenefitsjointmolecule,
      title={Synergistic Benefits of Joint Molecule Generation and Property Prediction}, 
      author={Adam Izdebski and Jan Olszewski and Pankhil Gawade and Krzysztof Koras and Serra Korkmaz and Valentin Rauscher and Jakub M. Tomczak and Ewa Szczurek},
      year={2025},
      eprint={2504.16559},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16559}, 
}
```

## References

[1] Brown, Nathan, et al. "GuacaMol: benchmarking models for de novo molecular design." Journal of chemical information and modeling, 2019.

[2] Zhou, Gengmo, et al. "Uni-mol: A universal 3d molecular representation learning framework." ICLR, 2023.