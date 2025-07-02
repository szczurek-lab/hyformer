""" Run the training script for the molecular property prediction task. """

import os, logging, sys

import torch

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel
from hyformer.trainers.trainer import Trainer

from hyformer.utils.reproducibility import set_seed

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
# logging.captureWarnings(False)

torch.set_float32_matmul_precision('high')

USE_DETERMINISTIC_ALGORITHMS = False


def main(
    out_dir: str,
    data_dir: str,
    experiment_seed: int,
    dataset_config_path: str,
    tokenizer_config_path: str,
    model_config_path: str,
    trainer_config_path: str,
    model_ckpt_path: str,
    split: str = 'test'
):    

    # Set seed
    set_seed(experiment_seed, use_deterministic_algorithms=USE_DETERMINISTIC_ALGORITHMS)

    # Load configurations
    dataset_config = DatasetConfig.from_config_filepath(dataset_config_path)
    tokenizer_config = TokenizerConfig.from_config_filepath(tokenizer_config_path)
    model_config = ModelConfig.from_config_filepath(model_config_path)
    trainer_config = TrainerConfig.from_config_filepath(trainer_config_path)
    
    # Initialize
    test_dataset = AutoDataset.from_config(dataset_config, split=split, root=data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model = AutoModel.from_config(
        model_config, prediction_task_type=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks
        )
    
    # Check for tokenizer and model vocabulary size mismatch
    assert len(tokenizer) == model.vocab_size, f"Tokenizer vocab size {len(tokenizer)} does not match model vocab size {model.vocab_size}"
    
    # Determine the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize trainer
    trainer = Trainer(
        config=trainer_config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        worker_seed=experiment_seed
        )

    trainer.resume_from_checkpoint(model_ckpt_path)
    
    # Run testing
    test_metric = trainer.test(
        test_dataset=test_dataset,
        metric=dataset_config.test_metric
        )
    
    return test_metric
