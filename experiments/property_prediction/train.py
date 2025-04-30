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
    model_ckpt_path: str = None,
    debug: bool = False,
    task_specific_validation: str = None,
    patience: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    max_epochs: int = None,
    weight_decay: float = None,
    decay_lr: bool = None,
    dropout: float = None
):    

    # Set seed
    set_seed(experiment_seed, use_deterministic_algorithms=USE_DETERMINISTIC_ALGORITHMS)

    # Load configurations
    dataset_config = DatasetConfig.from_config_filepath(dataset_config_path)
    tokenizer_config = TokenizerConfig.from_config_filepath(tokenizer_config_path)
    model_config = ModelConfig.from_config_filepath(model_config_path)
    trainer_config = TrainerConfig.from_config_filepath(trainer_config_path)
    
    # Collect configs that need to be updated and saved
    configs = [dataset_config, tokenizer_config, model_config, trainer_config]
        
    # Update all configs with command line arguments
    for config_file in configs:
        config_file.update(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            weight_decay=weight_decay,
            decay_lr=decay_lr,
            prediction_head_dropout_p=dropout
        )
    
    # Set debug mode
    if debug:
        model_config.num_transformer_layers = 2
        trainer_config.max_epochs = 2
        trainer_config.log_interval = 2
        trainer_config.warmup_iters = 200
    
    # Store configs within the output directory, for reproducibility
    if out_dir is not None:
        for config_file in configs:
            config_file.save(os.path.join(out_dir, f'{config_file.__class__.__name__}.json'))

    # Initialize
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model = AutoModel.from_config(
        model_config, prediction_task_type=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks
        )
    
    # Check for tokenizer and model vocabulary size mismatch
    assert len(tokenizer) == model.vocab_size, f"Tokenizer vocab size {len(tokenizer)} does not match model vocab size {model.vocab_size}"
    
    # Set debug mode
    if debug:
        train_dataset.data, train_dataset.target = train_dataset.data[:1500], train_dataset.target[:1500] if train_dataset.target is not None else None
        val_dataset.data, val_dataset.target = val_dataset.data[:1500], val_dataset.target[:1500] if val_dataset.target is not None else None
    
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

    if model_ckpt_path:
        trainer.resume_from_checkpoint(model_ckpt_path, discard_prediction_head=True)
        console.info(f"Resumed pre-trained model from {model_ckpt_path}")
    else:
        console.info("Training from scratch")
    
    # Run training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_specific_validation=task_specific_validation,
        patience=patience,
        )
    
    # Run testing
    val_metric = trainer.test(
        test_dataset=val_dataset,
        metric=dataset_config.test_metric
        )
    
    return val_metric
