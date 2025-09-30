# Hyformer

This repository is an optimized implementation of [Hyformer](https://arxiv.org/abs/2504.16559), a [joint](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LasserreBishopMinka06.pdf) transformer-based model that unifies a generative decoder with a predictive encoder. Depending on the task, Hyformer uses either a causal or a bidirectional mask and returns token probabilities or predicted property values.

<img src="_assets/hyformer.png" width="520" height="250"/>


## Installation

To create an environment that satisfies the necessary requirements run
```bash
pip install hyformer @ git+https://github.com/szczurek-lab/hyformer.git@v2.0 
```

## Pre-trained Models

Download pre-trained models from [HuggingFace](https://huggingface.co/SzczurekLab/hyformer):

- [hyformer_peptides_34M](https://huggingface.co/SzczurekLab/hyformer_peptides_34M) trained on 3.5M general-purpose and antimicrobial peptides.
- [hyformer_peptides_34M_MIC](https://huggingface.co/SzczurekLab/hyformer_peptides_34M_MIC) `Hyformer_peptides_34M` jointly fine-tuned on minimal inhibitory concentration values (MIC) against E. coli bacteria.

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
